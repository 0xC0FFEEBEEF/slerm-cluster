#!/usr/bin/env python3
"""slerm orchestrator + gateway (runs on slerm-head as a user service).

One stable OpenAI-compatible endpoint, http://slerm-head:8090/v1, in front of
whatever llama.cpp router is live. A request for a model that cannot be served
right now is HELD while the orchestrator brings up what that model needs
(nodes, workers, a router on the right device set), then forwarded. Idle
resources are released in tiers, and anything the user is doing outside slerm
(their own VMs, work on deadlock, batch jobs, a UPS event) wins.

    gateway     :8090   /v1/*  (held while bringing up), /v1/models (reachable models)
    control     127.0.0.1:8091  POST /yield /ups  (for hookscripts via ssh), GET /state
                                POST /stage /unstage  (the panel's Stage buttons)
    publishes   /data/llm/run/orchestrator.json   (the panel's model cards read it)

Power control is optional and switches on by itself when its credential exists:
    ~/.config/slerm-orch/pve.json       Proxmox API tokens (VM start/shutdown)
    ~/.config/slerm-orch/bmc-deadlock   deadlock BMC operator password
Without them the orchestrator still runs everything through Slurm and only
reports which node would have to be powered on.

Python 3.9 (the head's). The head cannot ssh to any node: everything here is
Slurm, HTTP to the routers, NFS files, and the hypervisors' HTTPS API.
"""
import http.client
import http.server
import json
import os
import re
import socketserver
import ssl
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request

sys.path.insert(0, "/data/llm/bin")
import slerm_topo  # noqa: E402
import thinking_caps  # noqa: E402  (2026-10-08: slerm.thinking in /v1/models, read by Open WebUI's thinking pill)

HOME = os.path.expanduser("~")
CONF_DIR = os.path.join(HOME, ".config/slerm-orch")
RUN = "/data/llm/run"
STATE_OUT = os.path.join(RUN, "orchestrator.json")
LOGF = os.path.join(HOME, "slerm-orch/orch.log")

CFG = {
    "gateway_port": 8090,
    "control_port": 8091,
    "tick": 15,
    # Tiered release (user's choice, 2026-09-27).
    "idle_release_s": 20 * 60,      # unload model + drop CPU workers
    "idle_poweroff_s": 60 * 60,     # stop router/GPU worker, power off what we started
    "bringup_timeout_s": 25 * 60,   # deadlock POST + VM boot + an 8-min model load
    "drain_wait_s": 10 * 60,        # max wait for in-flight requests before a router restart
    "router_port": 8080,
    "host_report_max_age": 120,
    "ups_max_age": 300,
    # Nodes the orchestrator may power off when idle. Head (slurmctld + NFS)
    # is never in this list.
    # Models that cannot do native tool calling: with a tool list in the prompt
    # llama3.2-1b answers EVERY message with a malformed call that llama.cpp's
    # parser rejects ("does not match the expected peg-native format", 6/6 on
    # 2026-09-27). The gateway drops tools/tool_choice for these.
    "no_tools_models": ["llama3.2-1b", "qwen2.5-coder-3b"],
    "managed_nodes": ["slerm-node01", "slerm-node02", "slerm-node04", "slerm-node05"],
    # Staging (user, 2026-09-29): at most this many models staged; staging one
    # more drops the oldest.
    "stage_max": 2,
    # Open WebUI's task model (titles, tags, follow-ups) since 2026-10-09: a CPU llama-server that router-rpc.sh
    # starts beside the router, on the router's node. The gateway never brings slerm up for it: with no router it
    # answers 503 at once (Open WebUI just skips the title). Long chats are trimmed to task_max_chars first.
    "task_model": "tasks-llama3.2-1b",
    "task_port": 8081,
    # 2026-10-21: the task model is its own on-demand job (jobs/tasks.sh) on one of these, first that is up;
    # cancelled after task_idle_s without a task request, and a node woken for it powers off again then.
    "task_nodes": ["slerm-node02", "slerm-node01"],
    "task_idle_s": 15 * 60,
    "task_wait_s": 240,             # a task request waits this long for a node to boot and the job to start
    "task_max_chars": 12000,
    # Model lock (2026-10-17): how often a locked model that went away may be brought back.
    "lock_restore_s": 10 * 60,
}
STAGED = os.path.join(RUN, "staged.json")

# Model lock (2026-10-17, user: "a 'lock model' button ... prevent the model from being unloaded, and prevent any other
# jobs from kicking the model off until unlocked, as well as prevent the nodes hosting it from powering off"). Set and
# cleared from the slerm panel (POST /lock, /unlock on the control port); kept in this file so it survives a restart.
# While locked: no idle unload/stop/shrink, no yielding the GPUs to waiting batch jobs (ComfyUI), the nodes hosting it
# never power off for idleness or by /poweroff, and requests for any other model are refused (409). The user's choices:
# their own VMs still win (slerm-yield hookscripts) and the UPS still wins; when the locked model goes away anyway it is
# brought back, at most once per lock_restore_s.
LOCK_FILE = os.path.join(RUN, "model-lock.json")


class LockedError(RuntimeError):
    """A model other than the locked one was asked for."""

# 2026-10-17 (user: "make a STRATA-* model selection in the model picker and wire it up"): models served by Strata
# (github.com/Niko1221/Strata, /data/llm/strata) instead of the llama.cpp router. One Strata server at a time, as Slurm
# job llm-strata (jobs/strata.sh) on node04's two P100s; it and the router exclude each other (both need both cards).
STRATA_REG = "/data/llm/etc/strata-models.json"
STRATA_JOB = "/data/llm/jobs/strata.sh"
TASK_JOB = "/data/llm/jobs/tasks.sh"
STRATA_TEMPLATE = "/data/llm/strata/Strata/serve/chat_template.jinja"
STRATA_READY = os.path.join(RUN, "strata-ready.json")   # jobs/strata.sh writes it once the n-gram table is warm

# The slerm machines. Head (300) is never touched: it is slurmctld and the NFS server.
VMS = {
    # Both Tesla P100s since 2026-10-04 (node03 retired, VM 303 kept but has no GPU).
    "slerm-node04": {"vmid": 304, "pve": "byteme", "mem_mb": 16384, "pci": ["0000:82:00", "0000:04:00"]},
    "slerm-node01": {"vmid": 301, "pve": "byteme2", "mem_mb": 12288},
    "slerm-node02": {"vmid": 302, "pve": "byteme2", "mem_mb": 12288},
}
PVE_HOSTS = {"byteme": "byteme", "byteme2": "byteme2"}
DEADLOCK_BMC = "deadlock-bmc"
MEM_MARGIN_MB = 1536
DRAIN_REASON = "slerm-orch"

# Candidate device sets, in preference order (fewest devices, GPUs first).
# rpc order must match router-rpc.sh: the GPU worker first, then CPU workers.
N1, N2, N4, N5 = "slerm-node01", "slerm-node02", "slerm-node04", "slerm-node05"
# node04 (both P100s) is the only GPU node. [3] must stay the full pool in
# slerm_topo.FULL_ORDER's order: plan_all() takes unreachable models' reasons from it.
CANDIDATES = [
    {"router": N4, "rpc": []},
    {"router": N4, "rpc": [N5]},
    {"router": N4, "rpc": [N1, N2]},
    {"router": N4, "rpc": [N1, N2, N5]},
    {"router": N5, "rpc": []},
    {"router": N5, "rpc": [N1, N2]},
]


def topo_devices(tid):
    """Nodes of a router from its topology id ("r-node04_node01-node02").
    A router we did not start (no id) is assumed to use the whole pool."""
    if not tid:
        return [N1, N2, N5, N4]
    router, _, rpc = tid[2:].partition("_")
    return ["slerm-" + x for x in rpc.split("-") if x and x != "none"] + ["slerm-" + router]


def model_nodes(devs, pm):
    """[{node, role, share %}] a model loads on: the router node (local device,
    last) plus every worker with a share of its tensor-split. Router first."""
    split = [float(x) for x in ((pm.get("keys") or {}).get("tensor-split") or "").split(",") if x.strip()]
    split += [0.0] * (len(devs) - len(split))
    total = sum(split) or 1.0
    # The split has one value per DEVICE: the rpc workers line up with devs, and
    # the router's local GPUs are everything after them.
    out = [{"node": devs[-1], "role": "router", "share": round(100 * sum(split[len(devs) - 1:]) / total)}]
    out += [{"node": n, "role": "gpu" if n in slerm_topo.GPU_NODES else "cpu",
             "share": round(100 * split[i] / total)} for i, n in enumerate(devs[:-1]) if split[i] > 0]
    return out


def is_strata(model):
    return isinstance(model, str) and model.startswith("STRATA-")


def strata_port():
    return int((read_json(STRATA_REG, {}) or {}).get("port", 18100))


def strata_registry():
    """{name: entry} for every STRATA-* registry entry whose files all exist (a model still downloading is not listed)."""
    reg = read_json(STRATA_REG, {}) or {}
    return {name: m for name, m in (reg.get("models") or {}).items()
            if is_strata(name) and all(os.path.exists(p) for p in m.get("requires", []))}


def strata_thinking(entry):
    """The thinking pill's options, from the template Strata renders (the same levels as llama.cpp's for Flash-Next)."""
    try:
        with open(entry.get("template") or STRATA_TEMPLATE, encoding="utf-8") as f:
            return thinking_caps.parse(f.read(), entry.get("reasoning_effort"))
    except (OSError, ValueError):
        return None


def log(*a):
    line = time.strftime("%Y-%m-%dT%H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    try:
        with open(LOGF, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def sh(cmd, timeout=30):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


# What a title, tags or follow-ups don't need from a chat: Code replies are mostly a folded steps list and a diff
# (<details> blocks), plus code. A 1B model fed that starts writing code instead of the JSON it was asked for.
TASK_NOISE = [(re.compile(r"<details\b.*?</details>", re.S), " "),
              (re.compile(r"```.*?```", re.S), " [code] "),
              (re.compile(r"\[(?:Preview|Fixer|Feature #\d+)\]\([^)]*\)"), "")]


def clean_task_prompt(j):
    for m in j.get("messages") or []:
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            for rx, repl in TASK_NOISE:
                m["content"] = rx.sub(repl, m["content"])
    return j


def trim_task_prompt(j, cap):
    """Open WebUI pastes the last few messages into one task prompt, and a Code reply alone can run to 50k
    characters, which a CPU model takes minutes to read. Cut the middle of each long message's chat history (the
    task's instructions before <chat_history> stay whole; a third of the room goes to the oldest part, which holds
    the user's question, the rest to the newest) until the prompt fits cap."""
    msgs = [m for m in j.get("messages") or [] if isinstance(m, dict) and isinstance(m.get("content"), str)]
    total = sum(len(m["content"]) for m in msgs)
    for m in sorted(msgs, key=lambda m: -len(m["content"])):
        if total <= cap:
            break
        c = m["content"]
        start = c.find("<chat_history>")
        start = start + len("<chat_history>") if start >= 0 else min(len(c), 1500)
        body = c[start:]
        keep = max(2000, len(body) - (total - cap))
        if keep >= len(body):
            continue
        head = keep // 3
        m["content"] = c[:start] + body[:head] + "\n[... trimmed ...]\n" + body[len(body) - (keep - head):]
        total -= len(c) - len(m["content"])
    return j


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    tmp = path + ".%d.tmp" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def http_json(url, method="GET", body=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


# -------------------------------------------------------------- actuators --

class PVE:
    """Proxmox API with a scoped token. Absent file = power control off."""

    def __init__(self):
        self.cfg = read_json(os.path.join(CONF_DIR, "pve.json"))
        # Self-signed certs on both hosts; the token is scoped to the slerm VMs.
        self.ctx = ssl._create_unverified_context()

    @property
    def enabled(self):
        return bool(self.cfg)

    def _req(self, host, path, method="GET", data=None):
        c = self.cfg[host]
        body = urllib.parse.urlencode(data).encode() if data else None
        req = urllib.request.Request(
            "https://%s:8006/api2/json%s" % (PVE_HOSTS[host], path), data=body, method=method,
            headers={"Authorization": "PVEAPIToken=%s=%s" % (c["tokenid"], c["secret"])})
        with urllib.request.urlopen(req, timeout=20, context=self.ctx) as r:
            return json.load(r)["data"]

    def host_state(self, host):
        node = self.cfg[host]["node"]
        st = self._req(host, "/nodes/%s/status" % node)
        vms = self._req(host, "/nodes/%s/qemu" % node)
        running = {}
        for v in vms:
            if v.get("status") == "running":
                cfg = self._req(host, "/nodes/%s/qemu/%s/config" % (node, v["vmid"]))
                pcis = [str(val).split(",")[0] for k, val in cfg.items() if k.startswith("hostpci")]
                running[int(v["vmid"])] = {"name": v.get("name"), "mem_mb": int(v.get("maxmem", 0)) // 2**20,
                                           "pci": pcis}
        mem = st["memory"]
        return {"avail_mb": int(mem.get("available", mem["free"])) // 2**20,
                "total_mb": int(mem["total"]) // 2**20, "running": running}

    def start(self, host, vmid):
        return self._req(host, "/nodes/%s/qemu/%d/status/start" % (self.cfg[host]["node"], vmid), "POST")

    def shutdown(self, host, vmid):
        return self._req(host, "/nodes/%s/qemu/%d/status/shutdown" % (self.cfg[host]["node"], vmid),
                         "POST", {"timeout": 120, "forceStop": 1})


class BMC:
    """deadlock's Intel BMC over IPMI-over-LAN.

    ipmitool unpacked from the Rocky RPM into ~/slerm-orch/pkg (the head has no
    sudo to install it). pyghmi was tried first and cannot talk to this BMC:
    the password is accepted but RAKP4 fails its integrity check."""

    IPMITOOL = os.path.join(HOME, "slerm-orch/pkg/usr/bin/ipmitool")
    VERBS = {"on": "on", "off": "off", "shutdown": "soft", "softoff": "soft"}

    def __init__(self):
        self.pwfile = os.path.join(CONF_DIR, "bmc-deadlock")

    @property
    def enabled(self):
        return os.path.exists(self.pwfile) and os.access(self.IPMITOOL, os.X_OK)

    def _run(self, *args):
        rc, out, err = sh([self.IPMITOOL, "-I", "lanplus", "-N", "3", "-R", "2", "-H", DEADLOCK_BMC,
                           "-U", "slerm", "-f", self.pwfile, "-L", "OPERATOR"] + list(args), timeout=30)
        if rc:
            raise RuntimeError((err or out).strip()[:200])
        return out

    def power(self):
        out = self._run("chassis", "power", "status").lower()
        return "on" if "is on" in out else "off" if "is off" in out else out.strip()

    def set(self, state):
        return self._run("chassis", "power", self.VERBS[state])


# ------------------------------------------------------------------ slurm --

def slurm_nodes():
    rc, out, _ = sh(["sinfo", "-N", "-h", "-o", "%N|%T|%E|%C|%e"])
    nodes = {}
    for line in out.splitlines():
        n, st, reason, cpus, freemem = line.split("|")
        a, i, o, t = (int(x) for x in cpus.split("/"))
        nodes[n] = {"state": st.lower(), "reason": reason, "cpu_alloc": a, "cpu_idle": i,
                    "cpu_total": t, "free_mem_mb": int(freemem) if freemem.isdigit() else 0}
    return nodes


def slurm_jobs():
    rc, out, _ = sh(["squeue", "-h", "-o", "%i|%j|%u|%T|%N|%P|%r|%b"])
    jobs = []
    for line in out.splitlines():
        i, name, user, st, nodes, part, reason, tres = line.split("|", 7)
        jobs.append({"id": i, "name": name, "user": user, "state": st, "nodes": nodes,
                     "partition": part, "reason": reason, "gpu": "gpu" in tres})
    return jobs


def node_up(sn):
    st = (sn or {}).get("state", "unknown")
    # sinfo %T appends "*" to a node that is not responding (e.g. "idle*").
    return "*" not in st and not any(x in st for x in ("down", "unknown", "not_responding", "fail", "power"))


def node_drained_by_us(sn):
    return sn and "drain" in sn.get("state", "") and sn.get("reason", "").startswith(DRAIN_REASON)


def scontrol(*args):
    rc, out, err = sh(["scontrol"] + list(args))
    if rc:
        log("scontrol", " ".join(args), "failed:", err.strip())
    return rc == 0


# ------------------------------------------------------------ orchestrator --

class Orchestrator:
    def __init__(self):
        self.lock = threading.RLock()
        self.ensure_lock = threading.Lock()
        self.freeze_lock = threading.Lock()
        self.pve = PVE()
        self.bmc = BMC()
        self.inflight = 0
        self.held = 0                # of inflight: gateway requests held for a bring-up (not being served)
        self.queue = {}              # model -> waiting count
        self.last_active = time.time()
        self.task_want = 0.0         # last task request that needed the on-demand task job (0 = none yet)
        self.task_woken = {}         # node -> when we woke it for the task model
        self.task_node = None        # node whose llm-tasks job is running
        self.metrics_prev = None
        self.started_by_us = set(read_json(os.path.join(CONF_DIR, "started.json"), []) or [])
        self.snapshot = {}
        self.yield_requests = []
        self.plans = {}
        self.staged = (read_json(STAGED, {}) or {}).get("models") or []
        self.stage_jobs = {}         # node -> llm-stage job id we submitted
        self.stage_woken = {}        # node -> time we powered it on for staging
        self.model_lock = read_json(LOCK_FILE, None) or None
        self.lock_restore_at = 0.0

    # ---- observation -------------------------------------------------------

    def ups(self):
        u = read_json(os.path.join(RUN, "ups.json"), {}) or {}
        age = time.time() - u.get("updated", 0)
        if age > CFG["ups_max_age"]:
            return {"known": False, "on_battery": False, "age_s": int(age)}
        return {"known": True, "on_battery": bool(u.get("on_battery") or u.get("powerfail")),
                "age_s": int(age)}

    # ---- power events --------------------------------------------------------

    # On battery every running job is frozen at once with `scontrol suspend`.
    # Under inference the GPUs otherwise keep drawing until ups-guardian's
    # shutdown reaches the nodes, 80+ s on 2026-10-01. Here suspend stops the
    # batch process and srun step tasks but NOT other children of the batch
    # script (tested 10-01; scancel --signal=STOP is worse). That covers
    # inference: the router execs llama-server, and the rpc workers only compute
    # when the router feeds them, so both GPUs go idle. Frozen, not
    # cancelled, so a blip that ups-guardian rides out costs nothing: utility
    # back -> resume. In a real outage the nodes are shut down anyway.
    def ups_changed(self, on_battery):
        path = os.path.join(RUN, "frozen_jobs.json")
        with self.freeze_lock:
            frozen = read_json(path, []) or []
            if on_battery:
                ids = [j["id"] for j in slurm_jobs()
                       if j["state"] == "RUNNING" and j["id"] not in frozen]
                done = [i for i in ids
                        if sh(["scontrol", "suspend", i], timeout=10)[0] == 0]
                if ids:
                    log("on battery: froze jobs", ",".join(done) or "-",
                        "(failed: %s)" % ",".join(sorted(set(ids) - set(done))) if len(done) < len(ids) else "")
                if done:
                    write_json(path, frozen + done)
            elif frozen:
                for i in frozen:
                    sh(["scontrol", "resume", i], timeout=10)
                log("utility back: resumed jobs", ",".join(frozen))
                write_json(path, [])

    def host_report(self, node="deadlock"):
        h = read_json(os.path.join(RUN, "host-%s.json" % node), {}) or {}
        if time.time() - h.get("updated", 0) > CFG["host_report_max_age"]:
            return None
        return h

    def router(self):
        """(job id, node, topology id) of the running router, from Slurm + its env."""
        for j in slurm_jobs():
            if j["name"] in ("llm-router", "llm-router-cpu") and j["user"] == os.environ.get("USER", "slerm"):
                return j
        return None

    def strata_job(self):
        """The Strata job (any state): {job, state, node, model}; the model is the job's comment."""
        rc, out, _ = sh(["squeue", "-h", "-u", os.environ.get("USER", "slerm"), "-n", "llm-strata",
                         "-o", "%i|%T|%N|%k"])
        for line in out.splitlines():
            i, st, node, comment = (line.split("|", 3) + ["", "", ""])[:4]
            return {"job": i, "state": st, "node": node, "model": comment.strip()}
        return None

    def strata_ready(self, job, node, port):
        """READY = the job has warmed the n-gram table (its marker names this job) and the server answers /health with
        a context (0 until the engine is up). Requests sent before the table warm finished ran at 4-8 tok/s."""
        if str((read_json(STRATA_READY, {}) or {}).get("job")) != str(job):
            return False
        try:
            h = http_json("http://%s:%d/health" % (node, port), timeout=5)
        except (OSError, ValueError):
            return False
        return bool(h.get("loaded") and (h.get("max_context") or 0) > 0)

    def router_models(self, node):
        try:
            d = http_json("http://%s:%d/models" % (node, CFG["router_port"]), timeout=5)
        except (OSError, ValueError):
            return None
        return {m["id"]: (m.get("status") or {}).get("value", "") for m in d.get("data", [])}

    def router_metrics_sig(self, node, loaded):
        sig = []
        for m in loaded:
            try:
                # autoload=false (2026-10-17): a model unloaded since the last look answers an error here instead
                # of being loaded again (the panel's /metrics?model= poll did that after every idle unload).
                with urllib.request.urlopen("http://%s:%d/metrics?model=%s&autoload=false" % (
                        node, CFG["router_port"], urllib.parse.quote(m)), timeout=5) as r:
                    txt = r.read().decode()
            except OSError:
                continue
            for key in ("requests_processing", "tokens_predicted_total", "prompt_tokens_total"):
                mm = re.search(r"llamacpp:%s\s+([0-9.e+]+)" % key, txt)
                if mm:
                    sig.append((m, key, float(mm.group(1))))
        return sig

    def observe(self):
        now = time.time()
        snap = {"updated": int(now), "ups": self.ups(), "slurm": slurm_nodes(), "jobs": slurm_jobs(),
                "pve": {}, "deadlock": {"host": self.host_report()}}
        if self.pve.enabled:
            for h in PVE_HOSTS:
                try:
                    snap["pve"][h] = self.pve.host_state(h)
                except Exception as e:                       # noqa: BLE001
                    snap["pve"][h] = {"error": str(e)}
        if self.bmc.enabled:
            try:
                snap["deadlock"]["power"] = self.bmc.power()
            except Exception as e:                           # noqa: BLE001
                snap["deadlock"]["power_error"] = str(e)
        r = self.router()
        snap["router"] = None
        if r and r["state"] == "RUNNING":
            models = self.router_models(r["nodes"])
            models = {} if models is None else models
            loaded = [m for m, s in models.items() if s in ("loaded", "loading")]
            snap["router"] = {"job": r["id"], "node": r["nodes"], "models": models, "loaded": loaded,
                              "topo": (read_json(os.path.join(RUN, "orch-router.json"), {}) or {}).get(
                                  str(r["id"]))}
            # /metrics?model= on a model that is still LOADING blocks until the load
            # finishes (the router logs "request cancelled while waiting"), so only
            # ask about models that are fully loaded.
            ready = [m for m, st in models.items() if st == "loaded"]
            sig = {(m, k): v for m, k, v in self.router_metrics_sig(r["nodes"], ready)}
            prev = self.metrics_prev or {}
            # Traffic that bypasses the gateway still counts: a counter that
            # moved for the SAME resident model. A router restart or unload
            # changes the key set, which is not activity.
            moved = any(k in prev and v > prev[k] for k, v in sig.items())
            if moved or any(s == "loading" for s in models.values()) or any(
                    k[1] == "requests_processing" and v > 0 for k, v in sig.items()):
                self.last_active = now
            self.metrics_prev = sig
        elif r:
            snap["router"] = {"job": r["id"], "node": r["nodes"], "state": r["state"]}
        if snap["router"]:
            # A router whose device set lost a node (powered off, crashed) still
            # lists its models, but every load would fail on the missing RPC
            # device. Mark it so the gateway re-plans instead of forwarding.
            devs = topo_devices(snap["router"].get("topo"))
            snap["router"]["degraded"] = [n for n in devs if not self.powered_up(n, snap)]
        s = self.strata_job()
        if s:
            s["port"] = strata_port()
            s["ready"] = s["state"] == "RUNNING" and self.strata_ready(s["job"], s["node"], s["port"])
            if not s["ready"]:
                self.last_active = now          # starting up counts as activity, like a router loading a model
        snap["strata"] = s
        if self.inflight or self.queue:
            self.last_active = now
        snap["idle_s"] = int(now - self.last_active)
        return snap

    def powered_up(self, n, snap):
        """Slurm up AND not known to be powered off. Slurm only notices a node
        that went away after SlurmdTimeout (300 s); the BMC / Proxmox know at once."""
        if not node_up(snap["slurm"].get(n)):
            return False
        if n == N5 and snap["deadlock"].get("power") == "off":
            return False
        if n in VMS:
            pv = snap["pve"].get(VMS[n]["pve"]) or {}
            if "running" in pv and VMS[n]["vmid"] not in pv["running"]:
                return False
        return True

    # ---- what can run where --------------------------------------------------

    def device_status(self, snap):
        """Per slerm node: up now, or startable, and why not."""
        out = {}
        battery = snap["ups"]["on_battery"]
        nodes = snap["slurm"]
        jobs = snap["jobs"]
        for n in (N1, N2, N4, N5):
            sn = nodes.get(n)
            d = {"up": self.powered_up(n, snap) and not ("drain" in (sn or {}).get("state", "") and not node_drained_by_us(sn)),
                 "startable": False, "reason": "", "drained_by_us": bool(node_drained_by_us(sn))}
            # Batch outranks LLM (partition PriorityTier 100 vs 1): a node busy
            # with someone's job is not available to a new LLM worker.
            # A CPU worker needs the whole node, so any batch job blocks it. On a
            # GPU node NEXRAD tasks use the spare cores by design; only a job
            # holding the GPU is in the way.
            # 2026-10-06: the on-demand ComfyUI worker (job name in lend_jobs) only BORROWS a node04 card and exits
            # after an idle spell, so it does not make node04 unavailable: an LLM request still plans onto node04
            # and its router job waits in Slurm for the card (the gateway holds the request) instead of falling
            # back to a CPU-only router on deadlock.
            lend = set(CFG.get("lend_jobs", ["comfyui"]))
            busy = [j for j in jobs if j["state"] == "RUNNING" and n in j["nodes"]
                    and not j["name"].startswith("llm-") and (n in (N1, N2, N5) or j["gpu"])
                    and not (n == N4 and j["name"] in lend)]
            if busy:
                d["batch_jobs"] = [j["id"] for j in busy]
            if n in VMS:
                v = VMS[n]
                pv = snap["pve"].get(v["pve"])
                if pv and "running" in pv:
                    running = pv["running"]
                    d["vm_running"] = v["vmid"] in running
                    d["booting"] = d["vm_running"] and not d["up"]
                    if not d["vm_running"]:
                        users = [r["name"] for vid, r in running.items()
                                 if any(p in r["pci"] for p in v.get("pci", [])) and vid != v["vmid"]]
                        if users:
                            d["reason"] = "P100s in use by %s" % ", ".join(users)
                        elif pv["avail_mb"] < v["mem_mb"] + MEM_MARGIN_MB:
                            d["reason"] = "%s short on RAM (%d MB free) for the VM" % (v["pve"], pv["avail_mb"])
                        else:
                            d["startable"] = not battery and self.pve.enabled
                elif not d["up"]:
                    d["reason"] = "VM is off (no Proxmox token configured)" if not self.pve.enabled \
                        else "cannot read %s" % v["pve"]
            else:   # deadlock
                host = snap["deadlock"].get("host")
                pw = snap["deadlock"].get("power")
                d["booting"] = pw == "on" and not d["up"]
                if not d["up"] and not d["booting"]:
                    if pw == "off":
                        d["startable"] = not battery
                        d["reason"] = "powered off"
                    else:
                        d["reason"] = "deadlock not responding" + ("" if self.bmc.enabled else
                                                                    " (no BMC credential configured)")
                if host:
                    d["host_busy"] = host.get("busy")
                    d["host_reasons"] = host.get("reasons", [])
                    d["cap_gb"] = max(0.0, host.get("free_for_slerm_mb", 0) / 1024.0 - 1.0)
            if battery and not d["up"]:
                d["reason"] = "UPS on battery"
            out[n] = d
        return out

    def usable(self, dev, n):
        d = dev[n]
        return (d["up"] or d["startable"] or d.get("booting")) and not d.get("batch_jobs")

    def topo_for(self, cand, dev):
        t = dict(cand)
        if N5 in slerm_topo.devices(t) and dev[N5].get("cap_gb") is not None:
            t["node05_cap_gb"] = min(slerm_topo.CPU_CAP_GB[N5], dev[N5]["cap_gb"]) \
                if t["router"] != N5 else dev[N5]["cap_gb"]
        return t

    def host_notes(self, dev):
        notes = []
        for n in (N4, N1, N2, N5):
            if not self.usable(dev, n):
                why = dev[n]["reason"] or ("busy with batch job(s) %s" % ",".join(dev[n].get("batch_jobs", [])))
                notes.append("%s: %s" % (n.replace("slerm-", ""), why))
        if dev[N5].get("host_busy"):
            notes.append("deadlock in use: " + "; ".join(dev[N5]["host_reasons"]))
        return ("; " + "; ".join(notes)) if notes else ""

    def plan_all(self, dev):
        """Best reachable topology per model, plus the plan for each candidate."""
        notes = self.host_notes(dev)
        best, plans, order = {}, {}, []
        for cand in CANDIDATES:
            devs = slerm_topo.devices(cand)
            if not all(self.usable(dev, n) for n in devs):
                continue
            t = self.topo_for(cand, dev)
            p, top, star = slerm_topo.plan(t, notes)
            plans[p["id"]] = (p, top, star, t)
            order.append(p)
        # A tested split on a usable device set beats a derived one on fewer
        # devices; derive only when nothing tested is reachable.
        for want_tested in (True, False):
            for p in order:
                for name, m in p["models"].items():
                    if m.get("feasible") and name not in best and (m.get("tested") or not want_tested):
                        best[name] = p["id"]
        # Reasons for the unreachable ones come from the widest candidate tried.
        full, _, _ = slerm_topo.plan(self.topo_for(CANDIDATES[3], dev), notes)
        models = {}
        for name, m in full["models"].items():
            if name in best:
                pm = plans[best[name]][0]["models"][name]
                models[name] = {"reachable": True, "topology": best[name], "reason": pm["reason"],
                                "tested": pm.get("tested", False), "est_tok_s": pm.get("est_tok_s"),
                                "nodes": model_nodes(plans[best[name]][0]["devices"], pm)}
            else:
                reason = m["reason"] if not m.get("feasible") else ("needs " + notes.lstrip("; "))
                models[name] = {"reachable": False, "reason": reason}
        return models, plans

    def strata_models(self, dev):
        """The STRATA-* models as model cards / picker entries: reachable whenever node04 is."""
        ok = self.usable(dev, N4)
        why = dev[N4]["reason"] or ("busy with batch job(s) %s" % ",".join(dev[N4].get("batch_jobs", [])))
        return {name: {"reachable": ok, "engine": "strata", "topology": "strata", "tested": True,
                       "reason": m.get("label", "Strata") if ok else "needs node04: " + why,
                       "est_tok_s": m.get("est_tok_s"), "nodes": [{"node": N4, "role": "router", "share": 100}]}
                for name, m in strata_registry().items()}

    # ---- actions -------------------------------------------------------------

    def save_started(self):
        try:
            write_json(os.path.join(CONF_DIR, "started.json"), sorted(self.started_by_us))
        except OSError:
            pass

    def bring_node_up(self, n, snap, dev):
        d = dev[n]
        sn = snap["slurm"].get(n)
        if d["drained_by_us"]:
            scontrol("update", "nodename=" + n, "state=resume")
        if d["up"]:
            return True
        if d.get("booting"):
            return False
        if not d["startable"]:
            raise RuntimeError("%s cannot be started: %s" % (n, d["reason"] or "unknown"))
        if n in VMS:
            v = VMS[n]
            log("starting VM", v["vmid"], "for", n)
            self.pve.start(v["pve"], v["vmid"])
        else:
            log("powering on deadlock via BMC")
            self.bmc.set("on")
        self.started_by_us.add(n)
        self.save_started()
        return False

    def wait_nodes(self, nodes, deadline):
        while time.time() < deadline:
            sn = slurm_nodes()
            if all(node_up(sn.get(n)) for n in nodes):
                for n in nodes:
                    if node_drained_by_us(sn.get(n)):
                        scontrol("update", "nodename=" + n, "state=resume")
                return True
            time.sleep(10)
        return False

    def stop_router(self, r):
        if not r:
            return
        log("cancelling router job", r["job"])
        sh(["scancel", str(r["job"])])
        for _ in range(60):
            if not self.router():
                break
            time.sleep(2)
        time.sleep(20)       # llama-server has no SO_REUSEADDR; TIME_WAIT blocks the bind

    def cancel_workers(self, keep_nodes):
        for j in slurm_jobs():
            if j["name"] in ("llm-rpcworker", "llm-rpccpu") and j["nodes"] not in keep_nodes:
                log("cancelling worker", j["id"], j["name"], "on", j["nodes"] or "(pending)")
                sh(["scancel", j["id"]])

    def start_router(self, p, top, star, t):
        ini, alw = slerm_topo.write_preset(p, top, star)
        env = dict(os.environ)
        cpu = [n for n in t["rpc"] if n not in slerm_topo.GPU_NODES]
        env.update({"LLM_CPU_WORKER": ",".join(cpu), "LLM_MODELS_PRESET": ini, "LLM_ALLOWED": alw,
                    "LLM_STRICT_DEVICES": "1"})
        env.pop("LLM_MODELS_DIR", None)
        args = ["sbatch", "--parsable", "--export=ALL"]
        env["LLM_GPU_WORKER"] = "none"       # no remote GPU any more: both P100s are local to node04
        if t["router"] == N4:
            args += ["--gres=gpu:pascal:2"]
        else:
            env["LLM_BUILD"] = "/data/llm/build-cpu-avx"
            args += ["--job-name=llm-router-cpu", "--partition=llmcpu", "--gres=none",
                     "--nodelist=" + N5, "--cpus-per-task=8", "--mem=28G"]
        args.append("/data/llm/jobs/router-rpc.sh")
        # Keep only the workers this topology uses; the router adopts live ones.
        self.cancel_workers(set(t["rpc"]) - {t["router"]})
        # A stage job would hold a core the CPU workers need (they take the
        # whole node); their own model-cache loops carry on staging. It fits
        # beside a router or GPU worker, so those nodes keep theirs.
        self.cancel_stage_jobs([n for n in slerm_topo.devices(t) if n not in slerm_topo.GPU_NODES])
        r = subprocess.run(args, capture_output=True, text=True, timeout=60, env=env)
        if r.returncode:
            raise RuntimeError("sbatch failed: " + r.stderr.strip())
        job = r.stdout.strip().split(";")[0]
        reg = read_json(os.path.join(RUN, "orch-router.json"), {}) or {}
        reg[job] = p["id"]
        write_json(os.path.join(RUN, "orch-router.json"), dict(list(reg.items())[-20:]))
        log("submitted router", job, "topology", p["id"], "devices", " ".join(p["devices"]))
        return job

    def wait_router(self, model, deadline):
        while time.time() < deadline:
            r = self.router()
            if r and r["state"] == "RUNNING":
                models = self.router_models(r["nodes"])
                if models and model in models:
                    return r["nodes"]
            elif r is None:
                raise RuntimeError("router job ended during start-up (see /data/llm/logs/router-*.out)")
            time.sleep(5)
        raise RuntimeError("router did not come up in time")

    def drain(self):
        """Wait (up to drain_wait_s) for requests that are being SERVED before taking the GPUs away. Requests held for
        a bring-up are waiting on this very bring-up and are not counted: counted (the old `inflight > 1`), two held
        requests made a switch sit out the whole drain_wait_s (2026-10-17 15:17 -> 15:27, the user's chat waiting)."""
        t_end = time.time() + CFG["drain_wait_s"]
        while time.time() < t_end:
            with self.lock:
                serving = self.inflight - self.held
            if serving <= 0:
                break
            time.sleep(3)

    def stop_strata(self, s):
        if not s:
            return
        log("cancelling strata job", s["job"], "(%s)" % (s.get("model") or "?"))
        sh(["scancel", str(s["job"])])
        for _ in range(60):
            if not self.strata_job():
                break
            time.sleep(2)

    def start_strata(self, model):
        env = dict(os.environ)
        env["STRATA_MODEL"] = model
        r = subprocess.run(["sbatch", "--parsable", "--export=ALL", "--comment=" + model, STRATA_JOB],
                           capture_output=True, text=True, timeout=60, env=env)
        if r.returncode:
            raise RuntimeError("sbatch failed: " + r.stderr.strip())
        job = r.stdout.strip().split(";")[0]
        log("submitted strata job", job, "for", model)
        return job

    def wait_strata(self, job, model, deadline):
        port = strata_port()
        while time.time() < deadline:
            s = self.strata_job()
            if not s or s["job"] != str(job):
                raise RuntimeError("Strata job %s ended during start-up (see /data/llm/logs/strata-%s.out)" % (job, job))
            if s["state"] == "RUNNING" and self.strata_ready(job, s["node"], port):
                log("strata", model, "ready on", s["node"], "(job %s)" % job)
                return s["node"]
            time.sleep(5)
        log("strata job", job, "not ready in time; cancelling it")
        sh(["scancel", str(job)])
        raise RuntimeError("Strata did not come up in time (see /data/llm/logs/strata-%s.out)" % job)

    def ensure_strata(self, model):
        """Make a STRATA-* model servable: stop the llama.cpp router (it holds both P100s), start jobs/strata.sh.
        Returns the node. Serialised with ensure()."""
        reg = strata_registry()
        if model not in reg:
            raise KeyError(model)
        deadline = time.time() + max(CFG["bringup_timeout_s"], int(reg[model].get("ready_s", 0)) + 900)
        with self.ensure_lock:
            self.check_lock(model)
            snap = self.observe()
            if snap["ups"]["on_battery"]:
                raise RuntimeError("slerm is on UPS battery; not starting anything")
            s = snap.get("strata")
            if s and s["model"] == model:
                if s.get("ready"):
                    return s["node"]
                log("strata job", s["job"], "still starting", model, "- waiting for it")
                return self.wait_strata(s["job"], model, deadline)
            dev = self.device_status(snap)
            if not self.usable(dev, N4):
                raise RuntimeError("%s cannot run right now: needs node04 (%s)" % (
                    model, dev[N4]["reason"] or "busy with batch job(s) %s" % ",".join(dev[N4].get("batch_jobs", []))))
            log("bringing up", model, "on Strata")
            if not self.bring_node_up(N4, snap, dev) and not self.wait_nodes([N4], deadline):
                raise RuntimeError("node04 did not come up")
            self.drain()
            if snap["router"]:
                self.stop_router(snap["router"])
            self.cancel_workers(set())
            self.stop_strata(s)
            job = self.start_strata(model)
            return self.wait_strata(job, model, deadline)

    def ensure(self, model):
        """Make MODEL servable. Returns the router node. Serialised."""
        self.check_lock(model)
        if is_strata(model):
            return self.ensure_strata(model)
        deadline = time.time() + CFG["bringup_timeout_s"]
        with self.ensure_lock:
            self.check_lock(model)          # locked while this request waited for the lock
            snap = self.observe()
            if snap["ups"]["on_battery"]:
                raise RuntimeError("slerm is on UPS battery; not starting anything")
            if snap.get("strata"):
                # A Strata server holds both P100s: a llama.cpp model needs them back.
                self.drain()
                self.stop_strata(snap["strata"])
                snap = self.observe()
            r = snap["router"]
            if r and r.get("models") and model in r["models"] and not r.get("degraded"):
                return r["node"]
            if r and r.get("degraded"):
                log("router", r["job"], "is missing", ",".join(r["degraded"]), "- re-planning")
            if r and not r.get("degraded") and r.get("state") in ("PENDING", "CONFIGURING") or (
                    r and r.get("models") is not None and not r["models"]):
                # A router that is still starting (workers, HELLO, bind) is not
                # answering yet. Give it time rather than restarting it.
                log("router", r["job"], "still starting; waiting for it")
                try:
                    return self.wait_router(model, min(deadline, time.time() + 600))
                except RuntimeError:
                    snap = self.observe()
                    r = snap["router"]
                    if r and r.get("models") and model in r["models"]:
                        return r["node"]
            dev = self.device_status(snap)
            models, plans = self.plan_all(dev)
            m = models.get(model)
            if not m:
                raise KeyError(model)
            if not m["reachable"]:
                raise RuntimeError("%s cannot run right now: %s" % (model, m["reason"]))
            p, top, star, t = plans[m["topology"]]
            log("bringing up", model, "on", p["id"])
            devs = slerm_topo.devices(t)
            waiting = [n for n in devs if not self.bring_node_up(n, snap, dev)]
            if waiting and not self.wait_nodes(waiting, deadline):
                raise RuntimeError("nodes did not come up: " + ", ".join(waiting))
            if r:
                self.drain()
                self.stop_router(r)
            self.start_router(p, top, star, t)
            return self.wait_router(model, deadline)

    def adopt(self, model=None):
        """Replace a router the orchestrator did not start (no topology) with a
        managed one, for MODEL (default: whatever it has resident)."""
        with self.ensure_lock:
            snap = self.observe()
            r = snap["router"]
            if r and r.get("topo"):
                return {"already": r["topo"]}
            model = model or next(iter((r or {}).get("loaded") or []), None)
            self.check_lock(model)
            dev = self.device_status(snap)
            models, plans = self.plan_all(dev)
            if model:
                tid = models[model]["topology"]
            else:
                tid = slerm_topo.topo_id(CANDIDATES[0])
            p, top, star, t = plans[tid]
            log("adopting router", r and r["job"], "->", tid, "for", model)
            self.stop_router(r)
            job = self.start_router(p, top, star, t)
        return {"router_job": job, "topology": tid, "model": model}

    def restart(self, model=None):
        """Fresh router AND workers (job-script changes only reach new jobs)."""
        with self.ensure_lock:
            snap = self.observe()
            r = snap["router"]
            model = model or next(iter((r or {}).get("loaded") or []), None)
            if self.model_lock and is_strata(self.model_lock.get("model")):
                raise LockedError("slerm is locked to %s: a router restart would stop it" % self.model_lock["model"])
            self.check_lock(model)
            dev = self.device_status(snap)
            models, plans = self.plan_all(dev)
            tid = models[model]["topology"] if model else slerm_topo.topo_id(CANDIDATES[0])
            p, top, star, t = plans[tid]
            log("restart (fresh workers) ->", tid, "for", model)
            self.stop_strata(snap.get("strata"))
            self.stop_router(r)
            self.cancel_workers(set())
            time.sleep(15)          # workers' :50052 TIME_WAIT
            job = self.start_router(p, top, star, t)
        return {"router_job": job, "topology": tid, "model": model}

    # ---- model lock ------------------------------------------------------------

    def check_lock(self, model):
        lk = self.model_lock
        if lk and model and model != lk.get("model"):
            raise LockedError("slerm is locked to %s (since %s): unlock it in the slerm panel's AI tab to use %s" % (
                lk.get("model"), time.strftime("%H:%M", time.localtime(lk.get("since") or 0)), model))

    def lock_nodes(self, snap):
        """The nodes the locked model runs on now; the ones it had when locked if it is down."""
        lk = self.model_lock
        if not lk:
            return set()
        model = lk.get("model")
        if is_strata(model):
            s = snap.get("strata") or {}
            if s.get("model") == model and s.get("node"):
                return {s["node"]}
        else:
            r = snap.get("router") or {}
            if (r.get("models") or {}).get(model) in ("loaded", "loading"):
                return set(topo_devices(r.get("topo"))) | ({r["node"]} if r.get("node") else set())
        return set(lk.get("nodes") or [])

    def set_lock(self, model=None):
        """Lock MODEL (default: whatever is loaded). Only a model that is loaded right now can be locked."""
        snap = self.observe()
        s = snap.get("strata") or {}
        r = snap.get("router") or {}
        serving = ([s["model"]] if s.get("model") and s.get("ready") else []) + \
            [m for m, st in (r.get("models") or {}).items() if st == "loaded"]
        model = model or (serving[0] if serving else None)
        if not model or model not in serving:
            raise ValueError("%s is not loaded: a model can be locked only while it is loaded" % (model or "nothing"))
        with self.lock:
            self.model_lock = {"model": model, "engine": "strata" if is_strata(model) else "llama.cpp",
                               "since": int(time.time())}
            self.model_lock["nodes"] = sorted(self.lock_nodes(snap))
            write_json(LOCK_FILE, self.model_lock)
        log("locked to", model, "(nodes %s)" % ",".join(self.model_lock["nodes"]))
        self.publish_lock(snap)
        return dict(self.model_lock)

    def publish_lock(self, snap):
        """Put the lock in orchestrator.json now rather than at the next tick, so the panel's button follows at once."""
        with self.lock:
            if not self.snapshot:
                return
            self.snapshot["lock"] = dict(self.model_lock) if self.model_lock else None
            self.snapshot["lock_nodes"] = sorted(self.lock_nodes(snap))
            out = dict(self.snapshot)
        try:
            write_json(STATE_OUT, out)
        except OSError as e:
            log("cannot publish state:", e)

    def clear_lock(self):
        with self.lock:
            lk, self.model_lock = self.model_lock, None
            # The idle tiers count from the unlock, not from the last request hours ago.
            self.last_active = time.time()
        try:
            os.remove(LOCK_FILE)
        except FileNotFoundError:
            pass
        log("unlocked", (lk or {}).get("model") or "(was not locked)")
        self.publish_lock({})
        return {"unlocked": (lk or {}).get("model")}

    def restore_locked(self, snap):
        """Bring the locked model back when it went away (crash, user VM, UPS), at most once per lock_restore_s."""
        lk = self.model_lock
        if not lk:
            return
        model = lk.get("model")
        if self.queue.get(model) or time.time() - self.lock_restore_at < CFG.get("lock_restore_s", 600):
            return
        if is_strata(model):
            st = snap.get("strata") or {}
            if st.get("model") == model:
                return                      # running, or still starting
        else:
            r = snap.get("router") or {}
            if (r.get("models") or {}).get(model) in ("loaded", "loading") and not r.get("degraded"):
                return
            if r.get("state") in ("PENDING", "CONFIGURING"):
                return
        if self.ensure_lock.locked():
            return                          # a bring-up is already running
        self.lock_restore_at = time.time()
        log("locked model", model, "is not running: bringing it back")

        def bring():
            with self.lock:
                self.queue[model] = self.queue.get(model, 0) + 1
            try:
                node = self.ensure(model)
                if not is_strata(model):
                    http_json("http://%s:%d/models/load" % (node, CFG["router_port"]), "POST",
                              {"model": model}, timeout=30)
            except Exception as e:                               # noqa: BLE001
                log("bringing back locked", model, "failed:", e)
            finally:
                with self.lock:
                    self.queue[model] -= 1
                    if self.queue[model] <= 0:
                        del self.queue[model]
        threading.Thread(target=bring, daemon=True, name="lock-restore").start()

    # ---- release ---------------------------------------------------------------

    def release(self, snap, dev):
        idle = snap["idle_s"]
        r = snap["router"]
        if self.inflight or self.queue or not self.ensure_lock.acquire(blocking=False):
            return
        try:
            if self.model_lock:
                # Locked: no unload, stop or shrink, no yielding the GPUs to waiting jobs. Only nodes the locked
                # model does not use may still power off at the idle tier.
                if idle >= CFG["idle_poweroff_s"]:
                    keep = self.lock_nodes(snap)
                    stopping = [n for n in CFG["managed_nodes"]
                                if n not in keep and dev[n]["up"] and not dev[n].get("batch_jobs")
                                and self.power_off_if_free(n, snap, dev, why="idle %d min" % (idle // 60),
                                                           wait=False)]
                    for n in stopping:
                        self.wait_powered_off(n)
                return
            # 2026-10-06: a non-llm job (ComfyUI) waits for node04's GPUs while the router holds them idle:
            # stop the router now instead of at the idle power-off. release() never runs mid-request.
            waiting = self.gpu_waiters(snap)
            s = snap.get("strata")
            if (r and r.get("node") == N4 or s) and waiting and idle >= CFG.get("gpu_yield_idle_s", 120):
                log("idle %d s: yielding %s GPUs to pending job(s) %s" % (idle, N4, ",".join(waiting)))
                self.stop_strata(s)
                if r and r.get("node") == N4:
                    self.stop_router(r)
                return
            # Strata has no "unload": the whole server goes at the first idle tier (it holds both cards and ~25-40 GB of
            # node04's RAM); the power-off tier then works as for the router.
            if s and idle >= CFG["idle_release_s"]:
                log("idle %d min: stopping Strata (%s)" % (idle // 60, s.get("model")))
                self.stop_strata(s)
                return
            if r and r.get("topo") and idle >= CFG["idle_release_s"] and idle < CFG["idle_poweroff_s"]:
                small = slerm_topo.topo_id(CANDIDATES[0])
                if r["topo"] != small and r.get("loaded") is not None:
                    # Nothing loaded is worth keeping: restart on the GPUs only,
                    # which frees node01/02/05 for batch and for the user.
                    cand = CANDIDATES[0]
                    if all(dev[n]["up"] for n in slerm_topo.devices(cand)):
                        t = self.topo_for(cand, dev)
                        p, top, star = slerm_topo.plan(t)
                        log("idle %d min: shrinking to GPU-only router" % (idle // 60))
                        self.stop_router(r)
                        self.start_router(p, top, star, t)
                        return
                for m in r.get("loaded") or []:
                    log("idle %d min: unloading %s" % (idle // 60, m))
                    try:
                        http_json("http://%s:%d/models/unload" % (r["node"], CFG["router_port"]),
                                  "POST", {"model": m})
                    except (OSError, ValueError) as e:
                        log("unload failed:", e)
            elif idle >= CFG["idle_poweroff_s"]:
                if r and r.get("topo"):
                    log("idle %d min: stopping router" % (idle // 60))
                    self.stop_router(r)
                    self.cancel_workers(set())
                stopping = [n for n in CFG["managed_nodes"]
                            if dev[n]["up"] and not dev[n].get("batch_jobs")
                            and self.power_off_if_free(n, snap, dev, why="idle %d min" % (idle // 60), wait=False)]
                for n in stopping:
                    self.wait_powered_off(n)
        finally:
            self.ensure_lock.release()

    def batch_waiting_for(self, n, snap):
        """Pending non-router batch jobs that would run on N (same test as wake_for_batch)."""
        pend = [j for j in snap["jobs"] if j["state"] == "PENDING" and not j["name"].startswith("llm-")]
        parts = set(p for j in pend for p in j["partition"].split(","))
        return bool((n in (N1, N2, N5) and parts & {"compute", "llmcpu"}) or (n == N4 and parts & {"gpu", "llm"}))

    def power_off_if_free(self, n, snap, dev, why, force=False, wait=True):
        if not force and n in self.lock_nodes(snap):
            return False            # hosts the locked model
        jobs = [j for j in snap["jobs"] if n in j["nodes"] and j["state"] in ("RUNNING", "COMPLETING")]
        if jobs and not force:
            return False
        if not force:
            # snap can be seconds old: a batch job that was woken for and just started on N is not in it yet.
            # 2026-10-07 the idle power-off shut node04 down under three such jobs (1 s after they started).
            if self.batch_waiting_for(n, snap):
                return False
            rc, out, _ = sh(["squeue", "-h", "-w", n, "-o", "%i"])
            if rc != 0 or out.strip():
                return False
        if n == N5 and dev[N5].get("host_busy") and not force:
            return False            # the user is using deadlock
        if n in VMS and not self.pve.enabled or n == N5 and not self.bmc.enabled:
            return False
        scontrol("update", "nodename=" + n, "state=drain", "reason=%s: %s" % (DRAIN_REASON, why))
        if force:
            for j in jobs:
                if j["name"].startswith("llm-"):
                    sh(["scancel", j["id"]])
                else:
                    sh(["scontrol", "requeue", j["id"]])
        log("powering off", n, "(%s)" % why)
        try:
            if n in VMS:
                self.pve.shutdown(VMS[n]["pve"], VMS[n]["vmid"])
            else:
                self.bmc.set("shutdown")
        except Exception as e:                                   # noqa: BLE001
            log("power off", n, "failed:", e)
            return False
        self.started_by_us.discard(n)
        self.save_started()
        if wait:
            self.wait_powered_off(n)
        return True

    def wait_powered_off(self, n, timeout=300):
        """Block until N is really off. Callers hold ensure_lock, so a request that
        arrives meanwhile queues and then sees the node off -- and starts it --
        instead of planning on a node that is still shutting down. 2026-10-04: a
        request 11 s after node04's idle shutdown got router 4433 submitted to the
        dying VM, which sat PENDING for the whole bring-up timeout (40 min)."""
        end = time.time() + timeout
        while time.time() < end:
            try:
                if n in VMS:
                    v = VMS[n]
                    if v["vmid"] not in self.pve.host_state(v["pve"])["running"]:
                        return True
                elif self.bmc.power() == "off":
                    return True
            except Exception as e:                               # noqa: BLE001
                log("power-off check of", n, "failed:", e)
            time.sleep(5)
        log("power off of", n, "not confirmed after %ds" % timeout)
        return False

    def gpu_waiters(self, snap):
        """Pending non-llm jobs held back by node04's GPUs (gpu/llm partitions, waiting on Resources)."""
        return [j["id"] for j in snap["jobs"] if j["state"] == "PENDING" and not j["name"].startswith("llm-")
                and set(j["partition"].split(",")) & {"gpu", "llm"} and "Resources" in j["reason"]]

    def wake_for_batch(self, snap, dev):
        """Pending batch jobs whose nodes we drained/powered off: bring them back."""
        pend = [j for j in snap["jobs"] if j["state"] == "PENDING" and not j["name"].startswith("llm-")
                and ("Resources" in j["reason"] or "NodeNotAvail" in j["reason"] or "Nodes_required" in j["reason"]
                     # 2026-10-20: with every node of the partition drained, squeue says "Nodes required for job are
                     # DOWN, DRAINED or reserved ..." (spaces, not underscores) and nothing woke for slerm-job.
                     or "Nodes required" in j["reason"])]
        if not pend or snap["ups"]["on_battery"]:
            return
        parts = set(p for j in pend for p in j["partition"].split(","))
        for n, d in dev.items():
            if d["up"] and not d["drained_by_us"]:
                continue
            in_part = (n in (N1, N2, N5) and parts & {"compute", "llmcpu"}) or \
                      (n == N4 and parts & {"gpu", "llm"})
            if in_part and (d["startable"] or d["drained_by_us"]):
                log("pending batch job(s) %s: waking %s" % (",".join(j["id"] for j in pend), n))
                try:
                    self.bring_node_up(n, snap, dev)
                except RuntimeError as e:
                    log(e)

    def handle_yield(self, req):
        """A user VM is about to start on a hypervisor: vacate the slerm VMs in its way."""
        host, need_mb, pcis = req.get("host"), int(req.get("mem_mb", 0)), req.get("pci", [])
        snap = self.observe()
        dev = self.device_status(snap)
        order = [n for n in (N4, N2, N1) if VMS[n]["pve"] == host]
        victims = []
        for n in order:
            if any(p in pcis for p in VMS[n].get("pci", [])):
                victims.append(n)
        pv = snap["pve"].get(host) or {}
        avail = pv.get("avail_mb", 0)
        for n in order:
            if avail >= need_mb + MEM_MARGIN_MB:
                break
            if n not in victims and (pv.get("running") or {}).get(VMS[n]["vmid"]):
                victims.append(n)
                avail += VMS[n]["mem_mb"]
        log("yield for %s on %s: stopping %s" % (req.get("vmid"), host, victims or "nothing"))
        with self.ensure_lock:
            r = snap["router"]
            if r and any(n in victims for n in topo_devices(r.get("topo"))):
                self.stop_router(r)
            if N4 in victims:
                self.stop_strata(snap.get("strata"))
            for n in victims:
                self.power_off_if_free(n, snap, dev, why="user VM %s" % req.get("vmid"), force=True)
        return {"stopped": victims}

    # ---- staging -------------------------------------------------------------------

    def save_staged(self):
        write_json(STAGED, {"cap": CFG["stage_max"], "updated": int(time.time()), "models": self.staged})

    def stage(self, model):
        """Put MODEL on the nodes it will load on, ahead of use (bin/model-cache.py)."""
        m = (self.snapshot.get("models") or {}).get(model)
        if m is None:
            raise KeyError(model)
        if is_strata(model):
            raise ValueError("STRATA models are not staged: they read the GGUFs from /data/llm/models and warm the "
                             "page cache when they start")
        if not m.get("reachable"):
            raise ValueError(m.get("reason") or "not reachable now")
        p = self.plans[m["topology"]][0]
        keys = p["models"][model].get("keys") or {}
        devs = list(p["devices"])
        split = [float(x) for x in (keys.get("tensor-split") or "").split(",") if x.strip()]
        split += [0.0] * (len(devs) - len(split))
        real = os.path.realpath(os.path.join(slerm_topo.SERVABLE, model + ".gguf"))
        entry = {"model": model, "at": int(time.time()), "topology": m["topology"], "devices": devs,
                 "split": split, "path": real, "bytes": os.path.getsize(real),
                 "nodes": [n for i, n in enumerate(devs) if i == len(devs) - 1 or split[i] > 0]}
        with self.lock:
            self.staged = [e for e in self.staged if e["model"] != model] + [entry]
            dropped = [e["model"] for e in self.staged[:-CFG["stage_max"]]]
            self.staged = self.staged[-CFG["stage_max"]:]
            self.save_staged()
            # The user is about to use slerm: restart the idle clock so what we
            # wake for staging is not powered straight back off.
            self.last_active = time.time()
        log("staged", model, "on", ",".join(entry["nodes"]), "(%s)" % m["topology"],
            ("dropped " + ",".join(dropped)) if dropped else "")
        return {"staged": model, "nodes": entry["nodes"], "dropped": dropped}

    def unstage(self, model):
        with self.lock:
            before = len(self.staged)
            self.staged = [e for e in self.staged if e["model"] != model]
            if len(self.staged) == before:
                raise KeyError(model)
            self.save_staged()
        log("unstaged", model)
        return {"unstaged": model}

    def cancel_stage_jobs(self, nodes):
        for n in nodes:
            job = self.stage_jobs.pop(n, None)
            if job:
                log("cancelling stage job", job, "on", n)
                sh(["scancel", job])

    def stage_status(self):
        """Per staged model: each node's progress from run/stage-<node>.json."""
        now = time.time()
        out = []
        for e in self.staged:
            nodes, done, total = {}, 0, 0
            for n in e["nodes"]:
                st = read_json(os.path.join(RUN, "stage-%s.json" % n), {}) or {}
                ms = (st.get("models") or {}).get(e["model"]) or {}
                fresh = ms.get("at") == e["at"]
                d = {"state": ms.get("state", "waiting") if fresh else "waiting",
                     "role": "router" if n == e["devices"][-1] else "worker",
                     "done": ms.get("done", 0) if fresh else 0, "total": ms.get("total") if fresh else None,
                     "note": ms.get("note", "") if fresh else "",
                     "stager_active": bool(st.get("active")) and now - (st.get("updated") or 0) < 90}
                nodes[n] = d
                done += d["done"] or 0
                total += d["total"] or 0
            states = set(d["state"] for d in nodes.values())
            overall = "done" if states == {"done"} else "no room" if "no room" in states else \
                "error" if "error" in states else "staging"
            out.append({"model": e["model"], "at": e["at"], "topology": e["topology"], "state": overall,
                        "done": done, "total": total, "nodes": nodes})
        return out

    def stage_tick(self, snap, dev, status):
        """Wake the nodes staging needs and give each one a stager."""
        jobs = {j["id"]: j for j in snap["jobs"]}
        for n, job in list(self.stage_jobs.items()):
            if job not in jobs:
                del self.stage_jobs[n]
        owed = set()
        for s in status:
            for n, d in s["nodes"].items():
                if d["state"] not in ("done", "no room", "error"):
                    owed.add(n)
        for n in sorted(owed):
            d = dev[n]
            if d.get("batch_jobs") or n in self.stage_jobs:
                continue
            if any(j["state"] == "RUNNING" and n in j["nodes"] and j["name"] in ("llm-rpccpu", "llm-router-cpu")
                   for j in snap["jobs"]):
                continue                # owns the whole node; its model-cache loop stages
            st = read_json(os.path.join(RUN, "stage-%s.json" % n), {}) or {}
            if st.get("active") and time.time() - (st.get("updated") or 0) < 90:
                continue                # a router/worker loop is already staging here
            if not d["up"]:
                if d["startable"] and time.time() - self.stage_woken.get(n, 0) > 900:
                    log("staging: waking", n)
                    self.stage_woken[n] = time.time()
                    try:
                        self.bring_node_up(n, snap, dev)
                    except RuntimeError as e:
                        log(e)
                continue
            if d["drained_by_us"]:
                scontrol("update", "nodename=" + n, "state=resume")
            part = "llm" if n in slerm_topo.GPU_NODES else "llmcpu"
            r = subprocess.run(["sbatch", "--parsable", "--partition=" + part, "--nodelist=" + n,
                                "/data/llm/jobs/stage.sh"], capture_output=True, text=True, timeout=60)
            if r.returncode:
                log("stage job for", n, "failed:", r.stderr.strip())
                continue
            self.stage_jobs[n] = r.stdout.strip().split(";")[0]
            log("submitted stage job", self.stage_jobs[n], "on", n)

    # ---- task model (Open WebUI titles, tags, follow-ups) -----------------------------

    def task_tick(self, snap, dev):
        """2026-10-21: the task model runs as its own job (llm-tasks, jobs/tasks.sh) on node02 or node01. It used to
        run beside Strata on node04, where its 1.6 GiB sat inside Strata's 41 GiB cap and left too little RAM to park
        a big conversation. A task request wakes a node for it (the gateway waits, task_wait_s); after task_idle_s
        without one the job is cancelled and a node woken for it powers off. A CPU router or worker waiting for a
        node comes first: the job is cancelled and the next task request starts it again."""
        now = time.time()
        jobs = [j for j in snap["jobs"] if j["name"] == "llm-tasks"]
        if jobs and not self.task_want:
            self.task_want = now                     # a job from before a restart: give it a full idle spell
        idle = now - self.task_want if self.task_want else float("inf")
        running = [j for j in jobs if j["state"] == "RUNNING"]
        self.task_node = running[0]["nodes"] if running else None
        cpu_wait = any(j["state"] == "PENDING" and j["name"] in ("llm-rpccpu", "llm-router-cpu") for j in snap["jobs"])
        if jobs and (idle > CFG["task_idle_s"] or cpu_wait):
            for j in jobs:
                log("cancelling task model job", j["id"], "(%s)" % (
                    "a CPU router is waiting for a node" if cpu_wait else "idle %d min" % (idle // 60)))
                sh(["scancel", j["id"]])
            self.task_node = None
            return
        if not jobs and idle <= CFG["task_idle_s"] and not cpu_wait:
            self.task_start(snap, dev)
        elif not jobs and idle > CFG["task_idle_s"]:
            for n, t in list(self.task_woken.items()):
                if not dev[n]["up"]:
                    if not dev[n].get("booting") and now - t > 600:
                        del self.task_woken[n]       # never came up, or someone else turned it off
                    continue
                if self.power_off_if_free(n, snap, dev, why="task model idle %d min" % (idle // 60), wait=False):
                    del self.task_woken[n]

    def task_start(self, snap, dev):
        """Submit the task job on the first task node that is up, else wake one and submit on a later tick."""
        cpu = set(j["nodes"] for j in snap["jobs"] if j["name"] in ("llm-rpccpu", "llm-router-cpu"))
        order = [n for n in CFG["task_nodes"] if n in dev and n not in cpu]
        for n in order:
            if dev[n]["up"]:
                if dev[n]["drained_by_us"]:
                    scontrol("update", "nodename=" + n, "state=resume")
                r = subprocess.run(["sbatch", "--parsable", "--nodelist=" + n, TASK_JOB],
                                   capture_output=True, text=True, timeout=60)
                if r.returncode:
                    log("task model job on", n, "failed:", r.stderr.strip())
                    continue
                log("submitted task model job", r.stdout.strip().split(";")[0], "on", n)
                return
        now = time.time()
        if any(dev[n].get("booting") or now - self.task_woken.get(n, 0) < 300 for n in order):
            return                                   # one is on its way
        for n in order:
            if dev[n]["startable"] or dev[n]["drained_by_us"]:
                log("task model: waking", n)
                self.task_woken[n] = now
                try:
                    self.bring_node_up(n, snap, dev)
                    return
                except RuntimeError as e:
                    log(e)
                    del self.task_woken[n]

    # ---- loop ----------------------------------------------------------------------

    def tick(self):
        snap = self.observe()
        dev = self.device_status(snap)
        models, plans = self.plan_all(dev)
        models.update(self.strata_models(dev))
        self.plans = plans
        staged = self.stage_status()
        with self.lock:
            self.snapshot = {"updated": snap["updated"], "ups": snap["ups"], "devices": dev,
                             "router": snap["router"], "strata": snap.get("strata"), "idle_s": snap["idle_s"],
                             "inflight": self.inflight, "queue": dict(self.queue),
                             "power_control": {"pve": self.pve.enabled, "bmc": self.bmc.enabled},
                             "started_by_us": sorted(self.started_by_us),
                             "deadlock": snap["deadlock"], "models": models,
                             "gateway": "http://slerm-head:%d/v1" % CFG["gateway_port"],
                             "staged": staged, "stage_max": CFG["stage_max"],
                             "lock": dict(self.model_lock) if self.model_lock else None,
                             "lock_nodes": sorted(self.lock_nodes(snap)),
                             "policy": {"idle_release_s": CFG["idle_release_s"],
                                        "idle_poweroff_s": CFG["idle_poweroff_s"]}}
        try:
            write_json(STATE_OUT, self.snapshot)
        except OSError as e:
            log("cannot publish state:", e)
        if snap["ups"]["on_battery"]:
            return          # ups-guardian owns the estate during a power event
        self.wake_for_batch(snap, dev)
        self.stage_tick(snap, dev, staged)
        self.task_tick(snap, dev)
        self.release(snap, dev)
        self.restore_locked(snap)

    def loop(self):
        while True:
            try:
                self.tick()
            except Exception:                                    # noqa: BLE001
                log("tick failed:", traceback.format_exc())
            time.sleep(CFG["tick"])


ORCH = Orchestrator()


# ------------------------------------------------------------------ gateway --

class Gateway(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "slerm-gateway/1"

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, obj, extra=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg, typ="slerm_unavailable"):
        self._send(code, {"error": {"message": msg, "type": typ, "code": code}})

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/v1/models", "/models"):
            snap = ORCH.snapshot
            data = []
            try:
                ini, paths = slerm_topo.parse_ini(), slerm_topo.servable()
            except Exception:                                # noqa: BLE001
                ini, paths = None, {}
            reg = strata_registry()
            st = snap.get("strata") or {}
            for name, m in sorted((snap.get("models") or {}).items()):
                if not m.get("reachable"):
                    continue
                if is_strata(name):
                    e = reg.get(name)
                    if e is None:
                        continue
                    status = ("loaded" if st.get("ready") else "loading") if st.get("model") == name else "cold"
                    info = {"status": status, "topology": "strata", "engine": "strata", "note": m.get("reason")}
                    if e.get("ctx"):
                        info["ctx"] = int(e["ctx"])
                    thinking = strata_thinking(e)
                    if thinking:
                        info["thinking"] = thinking
                    data.append({"id": name, "object": "model", "owned_by": "slerm", "slerm": info})
                    continue
                r = snap.get("router") or {}
                status = (r.get("models") or {}).get(name, "cold")
                info = {"status": status, "topology": m.get("topology"), "note": m.get("reason")}
                thinking = thinking_caps.caps(name, paths[name], ini) if ini and name in paths else None
                if thinking:
                    info["thinking"] = thinking
                # 2026-10-16: the context the router gives this model (its models.ini ctx-size, else [*]), for Open
                # WebUI's context ring and its auto-compact filter. Read from the file: asking the router (/props)
                # about a model that is not loaded makes it load that model.
                try:
                    sec = ini[2].get(name) if ini else None
                    if sec is not None:
                        info["ctx"] = int(sec.get("ctx-size") or ini[1].get("ctx-size") or 0) or None
                except (ValueError, TypeError, IndexError):
                    pass
                data.append({"id": name, "object": "model", "owned_by": "slerm", "slerm": info})
            lk = ORCH.model_lock
            if lk:
                for d in data:
                    if d["id"] == lk.get("model"):
                        d["slerm"]["locked"] = True
            # "sidecar", not "loaded": clients that look for the router's loaded model must not pick it.
            up = bool((snap.get("router") or {}).get("node") or ORCH.task_node)
            data.append({"id": CFG["task_model"], "object": "model", "owned_by": "slerm",
                         "slerm": {"status": "sidecar" if up else "cold", "topology": None,
                                   "note": "Open WebUI titles/tags/follow-ups; CPU beside the router, never wakes slerm"}})
            return self._send(200, {"object": "list", "data": data})
        if path in ("/health", "/"):
            return self._send(200, {"status": "ok", "router": (ORCH.snapshot.get("router") or {}).get("node"),
                                    "strata": (ORCH.snapshot.get("strata") or {}).get("model")})
        if path == "/slerm/state":
            return self._send(200, ORCH.snapshot)
        return self._proxy(None)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        return self._proxy(body)

    def _proxy(self, body):
        model = None
        stream = False
        if body:
            try:
                j = json.loads(body)
                model = j.get("model")
                stream = bool(j.get("stream"))
            except ValueError:
                pass
        # Clients configured with a connection prefix (Open WebUI) strip it
        # themselves; tolerate the old "slerm-turing." form anyway.
        if model and model.startswith("slerm-turing."):
            model = model.split(".", 1)[1]
            j["model"] = model
            body = json.dumps(j).encode()
        if model and model == CFG.get("task_model"):
            return self._task(j)
        lk = ORCH.model_lock
        if lk:
            asked = model
            if not asked and body is None:
                asked = (urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("model") or [None])[0]
            if asked and asked != lk.get("model") and asked != CFG.get("task_model"):
                try:
                    ORCH.check_lock(asked)
                except LockedError as e:
                    log("refused", asked, "(locked to %s)" % lk.get("model"))
                    return self._err(409, str(e), typ="slerm_locked")
        if model in CFG["no_tools_models"] and any(k in j for k in ("tools", "tool_choice", "functions")):
            for k in ("tools", "tool_choice", "parallel_tool_calls", "functions", "function_call"):
                j.pop(k, None)
            body = json.dumps(j).encode()
        # Per-client sampling profiles (2026-10-04). opencode cannot send top_k/min_p/presence_penalty (its per-model
        # "options" never reach the request body), so it tags requests "X-Slerm-Profile: coding" via a per-model
        # header and the gateway fills in that model's recommended values -- only keys the client did not set.
        # Open WebUI sends no header and gets the router presets' (chat) defaults.
        prof = (self.headers.get("X-Slerm-Profile") or "").strip().lower()
        if prof and model:
            want = ((CFG.get("sampling_profiles") or {}).get(prof) or {}).get(model) or {}
            added = {k: v for k, v in want.items() if k not in j}
            if added:
                j.update(added)
                body = json.dumps(j).encode()
        with ORCH.lock:
            ORCH.inflight += 1
            ORCH.last_active = time.time()
        sse_started = False
        try:
            port = None
            if is_strata(model):
                # STRATA-* (2026-10-17): the Strata server on node04, not the router
                st = ORCH.snapshot.get("strata") or {}
                ready = bool(st.get("model") == model and st.get("ready"))
                node = st.get("node")
                port = strata_port()
            else:
                r = ORCH.snapshot.get("router") or {}
                ready = model is None or (r.get("models") and model in r["models"] and not r.get("degraded"))
                node = r.get("node")
            if not ready:
                with ORCH.lock:
                    ORCH.queue[model] = ORCH.queue.get(model, 0) + 1
                    ORCH.held += 1
                result = {}

                def bring():
                    try:
                        result["node"] = ORCH.ensure(model)
                    except KeyError:
                        result["err"] = (404, "model %r is not offered by slerm" % model)
                    except LockedError as e:
                        result["err"] = (409, str(e))
                    except Exception as e:                   # noqa: BLE001
                        result["err"] = (503, str(e))
                th = threading.Thread(target=bring, daemon=True)
                th.start()
                # 2026-10-17: the queue entry goes in a finally. A client that hung up during the keep-alives raised
                # out of here before the decrement, and its entry stayed in ORCH.queue for good: idle_s read 0 from
                # then on, so nothing was ever released, yielded or powered off (found after the 01:17 restart).
                try:
                    if stream:
                        # Hold an SSE stream open with comments (ignored by clients)
                        # so nothing times out while nodes boot and the model loads.
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        sse_started = True
                        while th.is_alive():
                            self.wfile.write(b": slerm is bringing up " + model.encode() + b"\n\n")
                            self.wfile.flush()
                            th.join(10)
                    else:
                        th.join()
                finally:
                    with ORCH.lock:
                        ORCH.queue[model] -= 1
                        ORCH.held -= 1
                        if ORCH.queue[model] <= 0:
                            del ORCH.queue[model]
                if "err" in result:
                    code, msg = result["err"]
                    log("request for", model, "failed:", msg)
                    if sse_started:
                        self.wfile.write(b"data: " + json.dumps(
                            {"error": {"message": msg, "type": "slerm_unavailable"}}).encode() + b"\n\n")
                        return
                    return self._err(code, msg)
                node = result["node"]
            if not node:
                return self._err(503, "no slerm router is running")
            self._forward(node, body, sse_started, port=port)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with ORCH.lock:
                ORCH.inflight -= 1
                ORCH.last_active = time.time()

    @staticmethod
    def _task_node():
        """The node of a running, healthy llm-tasks job, waiting up to task_wait_s for task_tick to start one."""
        port = CFG.get("task_port", 8081)
        end = time.time() + CFG["task_wait_s"]
        while True:
            ORCH.task_want = time.time()
            n = ORCH.task_node
            if n:
                try:
                    c = http.client.HTTPConnection(n, port, timeout=3)
                    c.request("GET", "/health")
                    ok = c.getresponse().status == 200
                    c.close()
                    if ok:
                        return n
                except OSError:
                    pass
            if time.time() > end:
                return None
            time.sleep(2)

    def _task(self, j):
        """The task model. A router job runs it beside the router: straight there. Otherwise (Strata, or nothing up)
        it is the on-demand llm-tasks job on node02/node01 (2026-10-21): this request wakes it and waits up to
        task_wait_s. It does not count as activity, so titles never keep the GPUs or the router awake."""
        node = (ORCH.snapshot.get("router") or {}).get("node")
        if not node:
            node = self._task_node()
        if not node:
            return self._err(503, "the task model is starting on a slerm CPU node; try again in a minute")
        for k in ("tools", "tool_choice", "parallel_tool_calls", "functions", "function_call"):
            j.pop(k, None)
        before = sum(len(m.get("content") or "") for m in j.get("messages") or [] if isinstance(m.get("content"), str))
        clean_task_prompt(j)
        trim_task_prompt(j, CFG.get("task_max_chars", 12000))
        # Open WebUI's task prompts ask for one JSON object and drop anything else (the title then falls back to the
        # user's whole message). A 1B model answers in prose half the time: constrain it to JSON, short and steady.
        asked_json = any(isinstance(m.get("content"), str) and "json" in m["content"].lower() for m in j.get("messages") or [])
        if asked_json and not j.get("response_format"):
            # json_object is ignored by this llama.cpp build; a json_schema of any object constrains it (measured)
            j["response_format"] = {"type": "json_schema", "json_schema": {"name": "task", "schema": {"type": "object"}}}
        j["max_tokens"] = min(int(j.get("max_tokens") or 200), 200)
        j["temperature"] = 0.2
        after = sum(len(m.get("content") or "") for m in j.get("messages") or [] if isinstance(m.get("content"), str))
        if after < before:
            log("task prompt trimmed %d -> %d chars" % (before, after))
        try:
            self._forward(node, json.dumps(j).encode(), False, port=CFG.get("task_port", 8081), attempts=1)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError as e:
            log("task model on", node, "unreachable:", e)
            try:
                self._err(503, "task model on %s is not answering (%s)" % (node, e.__class__.__name__))
            except OSError:
                pass

    def _forward(self, node, body, sse_started, port=None, attempts=4):
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        # A request that lands while the router is unloading or swapping gets
        # its connection dropped with no response. Nothing has reached the
        # client yet, so retrying is safe.
        for attempt in range(attempts):
            conn = http.client.HTTPConnection(node, port or CFG["router_port"], timeout=3600)
            try:
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
                break
            except (http.client.RemoteDisconnected, ConnectionError, http.client.BadStatusLine) as e:
                conn.close()
                if attempt == attempts - 1:
                    raise
                log("router dropped the connection (%s); retrying" % e.__class__.__name__)
                time.sleep(3 * (attempt + 1))
        if not sse_started:
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
        elif resp.status != 200:
            self.wfile.write(b"data: " + json.dumps({"error": {"message": resp.read().decode(
                "utf-8", "replace")[:2000], "code": resp.status}}).encode() + b"\n\n")
            return
        while True:
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        self.close_connection = True
        conn.close()


class Control(http.server.BaseHTTPRequestHandler):
    """127.0.0.1 only: hookscripts and ups pushes arrive via forced-command ssh."""
    def log_message(self, fmt, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, ORCH.snapshot)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        if self.path == "/yield":
            try:
                return self._send(200, ORCH.handle_yield(req))
            except Exception as e:                               # noqa: BLE001
                log("yield failed:", traceback.format_exc())
                return self._send(500, {"error": str(e)})
        if self.path == "/lock":
            try:
                return self._send(200, ORCH.set_lock(req.get("model")))
            except ValueError as e:
                return self._send(409, {"error": str(e)})
        if self.path == "/unlock":
            return self._send(200, ORCH.clear_lock())
        if self.path == "/adopt":
            try:
                return self._send(200, ORCH.adopt(req.get("model")))
            except LockedError as e:
                return self._send(409, {"error": str(e)})
            except Exception as e:                               # noqa: BLE001
                log("adopt failed:", traceback.format_exc())
                return self._send(500, {"error": str(e)})
        if self.path == "/ensure":
            # The panel's "Bring up" button: start it and return at once. The
            # card follows progress through orchestrator.json.
            model = req.get("model")
            if model not in (ORCH.snapshot.get("models") or {}):
                return self._send(404, {"error": "unknown model %r" % model})
            if not ORCH.snapshot["models"][model].get("reachable"):
                return self._send(409, {"error": ORCH.snapshot["models"][model].get("reason")})
            try:
                ORCH.check_lock(model)
            except LockedError as e:
                return self._send(409, {"error": str(e)})

            def bring():
                with ORCH.lock:
                    ORCH.queue[model] = ORCH.queue.get(model, 0) + 1
                try:
                    node = ORCH.ensure(model)
                    if not is_strata(model):
                        http_json("http://%s:%d/models/load" % (node, CFG["router_port"]), "POST",
                                  {"model": model}, timeout=30)
                except Exception as e:                           # noqa: BLE001
                    log("ensure", model, "failed:", e)
                finally:
                    with ORCH.lock:
                        ORCH.queue[model] -= 1
                        if ORCH.queue[model] <= 0:
                            del ORCH.queue[model]
            threading.Thread(target=bring, daemon=True).start()
            return self._send(202, {"started": model})
        if self.path in ("/stage", "/unstage"):
            model = req.get("model")
            try:
                return self._send(200, ORCH.stage(model) if self.path == "/stage" else ORCH.unstage(model))
            except KeyError:
                return self._send(404, {"error": "unknown model %r" % model if self.path == "/stage"
                                        else "%r is not staged" % model})
            except ValueError as e:
                return self._send(409, {"error": str(e)})
            except Exception as e:                               # noqa: BLE001
                log("stage failed:", traceback.format_exc())
                return self._send(500, {"error": str(e)})
        if self.path == "/poweroff":
            # Manual test hook: drain + power off a managed node after a delay,
            # through the same path the idle release uses (force: jobs requeued).
            node, delay = req.get("node"), int(req.get("delay_s", 0))
            if node not in CFG["managed_nodes"]:
                return self._send(400, {"error": "not a managed node"})
            if node in ORCH.lock_nodes(ORCH.observe()):
                return self._send(409, {"error": "%s hosts the locked model %s" % (node, ORCH.model_lock["model"])})

            def later():
                time.sleep(delay)
                snap = ORCH.observe()
                ORCH.power_off_if_free(node, snap, ORCH.device_status(snap), why="manual power-off", force=True)
            threading.Thread(target=later, daemon=True).start()
            log("scheduled power-off of", node, "in", delay, "s")
            return self._send(202, {"node": node, "in_s": delay})
        if self.path == "/restart":
            try:
                return self._send(200, ORCH.restart(req.get("model")))
            except LockedError as e:
                return self._send(409, {"error": str(e)})
            except Exception as e:                               # noqa: BLE001
                log("restart failed:", traceback.format_exc())
                return self._send(500, {"error": str(e)})
        if self.path == "/ups":
            req["updated"] = int(time.time())
            write_json(os.path.join(RUN, "ups.json"), req)
            try:
                ORCH.ups_changed(bool(req.get("on_battery") or req.get("powerfail")))
            except Exception:                                    # noqa: BLE001
                log("ups freeze/thaw failed:", traceback.format_exc())
            return self._send(200, {"ok": True})
        self._send(404, {"error": "unknown"})


class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    # The unit and this file live in the NFS-shared home, so every node's user
    # manager can see them. Only the head may run the orchestrator: six copies
    # would fight over the same router (happened on first install, 2026-09-27).
    import socket
    if not socket.gethostname().startswith("slerm-head"):
        sys.exit("slerm-orch runs only on slerm-head, not " + socket.gethostname())
    cfg = read_json(os.path.join(CONF_DIR, "config.json"), {}) or {}
    CFG.update(cfg)
    log("starting; power control pve=%s bmc=%s" % (ORCH.pve.enabled, ORCH.bmc.enabled))
    threading.Thread(target=ORCH.loop, daemon=True, name="tick").start()
    threading.Thread(target=Threaded(("127.0.0.1", CFG["control_port"]), Control).serve_forever,
                     daemon=True, name="control").start()
    Threaded(("0.0.0.0", CFG["gateway_port"]), Gateway).serve_forever()


if __name__ == "__main__":
    main()
