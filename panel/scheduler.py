"""In-app scheduler.

The cluster has `scrontab` disabled, so recurrence lives here instead: a
daemon thread ticks every few seconds and fires due schedules through exactly
the same code path as a manual run.
"""
import threading
import time

from croniter import croniter

import config
import db
import runs

_thread = None
_stop = threading.Event()


class ScheduleError(ValueError):
    pass


def validate(kind, expr):
    if kind == "cron":
        if not croniter.is_valid(expr):
            raise ScheduleError("not a valid cron expression: %r" % expr)
    elif kind == "interval":
        try:
            seconds = int(expr)
        except (TypeError, ValueError):
            raise ScheduleError("interval must be a number of seconds")
        if seconds < 60:
            raise ScheduleError("interval must be at least 60 seconds")
    else:
        raise ScheduleError("kind must be 'cron' or 'interval'")
    return True


def next_fire(kind, expr, base=None):
    base = base or time.time()
    if kind == "cron":
        return croniter(expr, base).get_next(float)
    return base + int(expr)


def describe(kind, expr):
    if kind == "interval":
        seconds = int(expr)
        if seconds % 3600 == 0:
            return "every %d hour(s)" % (seconds // 3600)
        if seconds % 60 == 0:
            return "every %d minute(s)" % (seconds // 60)
        return "every %d seconds" % seconds
    return "cron: %s" % expr


def fire(schedule, force=False):
    """Run a schedule now. Returns the run id, or None if it was skipped."""
    if not force:
        active = db.active_runs(schedule["project"])
        if active:
            db.update_schedule(schedule["id"], last_fired=time.time(),
                               last_error="skipped: run %d still active"
                                          % active[0]["id"])
            return None
    try:
        run_id = runs.start(schedule["project"], schedule["params"],
                            schedule["options"], source="schedule",
                            schedule_id=schedule["id"])
    except Exception as exc:  # noqa: BLE001 - keep the tick loop alive
        db.update_schedule(schedule["id"], last_fired=time.time(),
                           last_error="%s: %s" % (type(exc).__name__, exc))
        return None
    db.update_schedule(schedule["id"], last_fired=time.time(),
                       last_run_id=run_id, last_error=None)
    return run_id


def tick(now=None):
    now = now or time.time()
    fired = []
    for schedule in db.list_schedules():
        if not schedule["enabled"]:
            continue
        due = schedule["next_fire"]
        if due is None:
            db.update_schedule(schedule["id"],
                               next_fire=next_fire(schedule["kind"], schedule["expr"], now))
            continue
        if due > now:
            continue
        run_id = fire(schedule)
        if run_id:
            fired.append((schedule["id"], run_id))
        # advance past any missed windows so a stopped panel does not stampede
        upcoming = next_fire(schedule["kind"], schedule["expr"], now)
        db.update_schedule(schedule["id"], next_fire=upcoming)
    return fired


def _loop():
    while not _stop.wait(config.TICK_SECONDS):
        try:
            tick()
        except Exception:  # noqa: BLE001 - never let the loop die
            import traceback
            traceback.print_exc()


def start():
    global _thread
    if _thread and _thread.is_alive():
        return _thread
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="scheduler", daemon=True)
    _thread.start()
    return _thread


def stop():
    _stop.set()


def status():
    return {
        "running": bool(_thread and _thread.is_alive()),
        "tick_seconds": config.TICK_SECONDS,
    }
