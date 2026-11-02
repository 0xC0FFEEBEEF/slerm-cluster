#!/usr/bin/env python3
"""Raise the chassis minimum fan PWM via iDRAC when the slerm GPUs get hot.

WHY A SINGLE CONTROLLER
Both GPU nodes are VMs on the same physical host, so there is exactly ONE set
of fans. If each node drove them they would fight, the cooler one continually
cancelling the hotter one. This runs in ONE place (the head), reads the
telemetry BOTH nodes already publish to /data/llm/run, and decides from the
hottest card. No locking is needed because there is only ever one writer.

WHY MinimumFanSpeed
iDRAC cannot see the GPUs at all -- its sensors are Inlet, Exhaust, CPU1, CPU2 --
so its automatic curve is blind to the cards we actually need to cool.
MinimumFanSpeed raises a FLOOR (10-100%); the automatic curve still runs on top
and will go higher if the CPUs demand it. We only ever raise the floor, never
cap cooling, so the failure mode is benign: if this process dies the fans revert
to normal automatic behaviour. 255 means "no floor", i.e. pure automatic.

FanSpeedOffset is deliberately left alone.

WHY racadm AND NOT REDFISH
This is an iDRAC8 / 13G (firmware 2.83). Redfish here is read-only for
thermals: the Attributes endpoint that carries these settings on iDRAC9 returns
404. racadm over SSH does expose them. Auth is by SSH key (racadm sshpkauth),
so no password is stored anywhere.

Safety rules:
  - exiting for any reason restores 255 (pure automatic control)
  - stale or missing telemetry also restores 255; unknown temps mean deferring
    to the iDRAC own thermal management, which runs underneath regardless
  - --apply is required to write anything; the default is a dry run
"""
import argparse, json, os, signal, subprocess, sys, time

RUN_DIR = "/data/llm/run"
MAX_AGE = 30.0     # telemetry older than this is treated as absent
DWELL = 90.0       # a cooler reading must hold this long before stepping down
POLL = 15.0        # racadm over ssh is slow; no point hammering it
QUANTUM = 5        # only move the floor in 5% steps, to limit writes

# Idle curve on the hottest GPU: quiet. Below TEMP_LO there is no floor at all.
TEMP_LO, PWM_LO = 60.0, 20     # first floor once a card warms up
TEMP_HI, PWM_HI = 90.0, 100    # full send

# Busy curve (2026-09-26): while any card is doing inference, cool hard and early.
# Busy = util_pct only. Power is useless as a signal: the 980 Ti draws ~105 W at 0%
# util whenever a model is resident (it never leaves P2 with a CUDA context open).
BUSY_UTIL = 10
BUSY_HOLD = 120.0              # stay in busy mode this long after the last busy sample
BUSY_TEMP_LO, BUSY_PWM_LO = 50.0, 70   # busy floor starts at 70%
BUSY_TEMP_HI = 65.0                    # 100% from here up

# Any single card pegged -> full chassis fans, whatever the temps say. Also never
# run the chassis slower than the busiest GPU's own fan, and go full once a GPU
# fan is working hard (fan_pct needs a gpu-publish.py from 2026-09-26 or later;
# older publishers simply omit it).
FULL_UTIL = 90
FULL_GPU_FAN = 75
AUTO = 255                     # racadm value meaning "no minimum"
PWM_FLOOR, PWM_CEIL = 10, 100  # MFSMinimumLimit / MFSMaximumLimit on this box

ATTR = "system.thermalsettings.MinimumFanSpeed"


def read_temps():
    """Hottest GPU across all nodes with fresh telemetry, plus per-node detail."""
    now, hottest, detail = time.time(), None, []
    busy, full, gpu_fan = False, False, 0
    try:
        names = os.listdir(RUN_DIR)
    except OSError:
        return None, detail, busy, full, gpu_fan
    for fn in names:
        if not (fn.startswith("gpu-") and fn.endswith(".json")):
            continue
        try:
            with open(os.path.join(RUN_DIR, fn)) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        ts = doc.get("ts")
        if not isinstance(ts, (int, float)) or now - ts > MAX_AGE:
            continue
        for g in doc.get("gpus", []):
            t = g.get("temp_c")
            if t is None:
                continue
            util, power = g.get("util_pct") or 0, g.get("power_w") or 0
            fan = g.get("fan_pct")
            detail.append((doc.get("host"), g.get("name"), t, util, power, fan))
            hottest = t if hottest is None else max(hottest, t)
            busy = busy or util >= BUSY_UTIL
            full = full or util >= FULL_UTIL or (fan or 0) >= FULL_GPU_FAN
            gpu_fan = max(gpu_fan, fan or 0)
    return hottest, detail, busy, full, gpu_fan


def target_pwm(temp, busy=False):
    """Hottest GPU temp -> minimum fan PWM, or AUTO for no floor."""
    if temp is None:
        return AUTO
    if busy:
        lo, plo, hi = BUSY_TEMP_LO, BUSY_PWM_LO, BUSY_TEMP_HI
        temp = max(temp, lo)            # busy never drops below the busy floor
    else:
        lo, plo, hi = TEMP_LO, PWM_LO, TEMP_HI
    if temp < lo:
        return AUTO
    if temp >= hi:
        return PWM_HI
    span = (temp - lo) / (hi - lo)
    pwm = plo + span * (PWM_HI - plo)
    pwm = int(round(pwm / QUANTUM) * QUANTUM)          # quantise to limit writes
    return max(PWM_FLOOR, min(PWM_CEIL, pwm))


class Racadm:
    def __init__(self, host, user, key):
        self.target = "%s@%s" % (user, host)
        self.key = key

    def _run(self, cmd, timeout=90):
        args = ["ssh", "-i", self.key, "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=%s" % os.path.expanduser("~/.ssh/known_hosts_idrac"),
                "-o", "ConnectTimeout=20", self.target, cmd]
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)

    def get_pwm(self):
        # iDRAC8 ssh is flaky under rapid reconnects; a couple of retries is
        # cheaper than dying and letting systemd restart the whole controller.
        for attempt in range(3):
            r = self._run("racadm get %s" % ATTR)
            for line in r.stdout.splitlines():
                if line.strip().startswith("MinimumFanSpeed="):
                    try:
                        return int(line.split("=", 1)[1].strip())
                    except ValueError:
                        return AUTO
            time.sleep(5)
        raise RuntimeError("could not read %s" % ATTR)

    def set_pwm(self, value):
        r = self._run("racadm set %s %d" % (ATTR, value))
        blob = (r.stdout + r.stderr).lower()
        if "success" not in blob:
            raise RuntimeError((r.stdout + r.stderr).strip()[:200])
        return True


def is_cooling(current, want):
    """True when the change reduces cooling, which is the direction we delay."""
    if current == AUTO:
        return False          # anything away from "no floor" is a step up
    if want == AUTO:
        return True           # dropping the floor entirely
    return want < current


def label(v):
    return "auto (no floor)" if v == AUTO else "%d%%" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("IDRAC_HOST", "idrac"))
    ap.add_argument("--user", default=os.environ.get("IDRAC_USER", "root"))
    ap.add_argument("--key", default=os.environ.get(
        "IDRAC_SSH_KEY", os.path.expanduser("~/.ssh/id_rsa_idrac")))
    ap.add_argument("--apply", action="store_true", help="actually change the floor")
    ap.add_argument("--once", action="store_true", help="evaluate once and exit")
    args = ap.parse_args()

    drac = Racadm(args.host, args.user, args.key)
    try:
        current = drac.get_pwm()
    except Exception as e:
        print("iDRAC unreachable: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    print("connected; %s = %s" % (ATTR, label(current)), flush=True)

    state = {"applied": current, "since": 0.0, "stop": False, "busy_at": 0.0}

    def restore(*_):
        state["stop"] = True
        if args.apply and state["applied"] != AUTO:
            try:
                drac.set_pwm(AUTO)
                print("restored minimum fan speed to auto", flush=True)
            except Exception as e:
                print("WARNING could not restore: %s" % e, file=sys.stderr)
    signal.signal(signal.SIGTERM, restore)
    signal.signal(signal.SIGINT, restore)

    try:
        while not state["stop"]:
            temp, detail, busy_now, full, gpu_fan = read_temps()
            if busy_now:
                state["busy_at"] = time.time()
            busy = time.time() - state["busy_at"] < BUSY_HOLD
            want = target_pwm(temp, busy)
            if temp is not None:
                if full:
                    want = PWM_HI
                elif gpu_fan:
                    floor = min(PWM_CEIL, -(-gpu_fan // QUANTUM) * QUANTUM)
                    want = floor if want == AUTO else max(want, floor)
            mode = "  [FULL]" if full else "  [busy]" if busy else ""
            for host, name, t, util, power, fan in sorted(detail):
                print("  %-14s %-26s %sC util %s%% %sW fan %s%s" % (
                    host, name, t, util, power,
                    "-" if fan is None else "%s%%" % fan, mode), flush=True)
            if temp is None:
                print("no fresh telemetry -> deferring to automatic control", flush=True)

            if want != state["applied"]:
                # Raising the floor happens at once; lowering it waits out DWELL so
                # the fans do not surge and drop as a card flickers around a step.
                if not (is_cooling(state["applied"], want)
                        and time.time() - state["since"] < DWELL):
                    print("hottest %sC: %s -> %s%s" % (
                        temp, label(state["applied"]), label(want),
                        "" if args.apply else "  [DRY RUN]"), flush=True)
                    if args.apply:
                        try:
                            drac.set_pwm(want)
                        except Exception as e:
                            print("WARNING set failed: %s" % e, file=sys.stderr)
                        else:
                            state["applied"] = want
                            state["since"] = time.time()
                    else:
                        state["applied"] = want
                        state["since"] = time.time()
            if args.once:
                break
            time.sleep(POLL)
    finally:
        restore()
    return 0


if __name__ == "__main__":
    sys.exit(main())
