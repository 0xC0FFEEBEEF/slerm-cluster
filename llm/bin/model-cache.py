#!/usr/bin/env python3
"""Node-local model cache for the slerm llama.cpp router and its RPC workers.

Loading a model off NFS is the slow part of every swap: ~57 MB/s under load
(217 MB/s at best) against ~586 MB/s from a node's own disk, so a 23 GB model
takes ~8 min cold. This keeps the most recently used models on local disk.

Policy (the user's): keep the last KEEP models used, and forget a model once it
has not been used for TTL seconds. "Used" means resident in the router.

  router mode  (on the router's node, started by router-rpc.sh)
    The router is pointed at a node-local LINK FARM instead of
    models/servable. Each farm link resolves to the local copy when there is
    one, otherwise to the NFS file, and llama.cpp opens the path afresh on
    every load, so a link swap takes effect on the next load with no restart.
    After a model has finished loading it is copied here in the background,
    throttled, and paused while any load is running so it never competes with
    one for NFS. Eviction swaps the link back to NFS first, then deletes; a
    running server keeps its mmap of a deleted file, so that is always safe.
    Publishes /data/llm/run/model-activity.json (load windows + keep set).

  worker mode  (on each RPC worker, started by the rpc-worker job scripts)
    ggml-rpc-server --cache stores every tensor over 10 MB it receives as
    $LLAMA_CACHE/rpc/<fnv hash>; on the next load the router sends the hash
    only and the worker reads its own disk instead of the network. The files
    carry no model name, so each new file is attributed to the model(s) the
    router was loading when it was written, and evicted with that model.

  init-farm    build the link farm once, synchronously, and print its path.

  stage        (a Slurm job the orchestrator submits, "llm-stage") run the
    stager until this node holds everything staged for it, then exit.

Staging (the panel's Stage button): /data/llm/run/staged.json, written by
slerm-orch, lists up to 2 models the user is about to use, each with the
device set and tensor-split it will load on. The stager puts a full local copy
on that model's router node and pre-writes the worker's rpc cache files (the
exact tensors it will be sent -- bin/model_stage.py) on every worker with a
share. It pauses while any model is loading. Staged models count as kept on
every node until they are unstaged or pushed out by a newer stage. It runs in
the llm-stage job, and also as a thread of the router/worker loops, whichever
takes the node's stage.lock first; status goes to run/stage-<node>.json.

Both modes publish /data/llm/run/cache-<node>.json for the panel. Python 3.9
(the nodes' version): no match statements, no X | Y type unions.
"""
import fcntl
import json
import os
import shutil
import socket
import sys
import threading
import time
import subprocess
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_stage  # noqa: E402

CACHE = os.environ.get("LLM_LOCAL_CACHE", "/var/cache/slerm-llm")
SERVABLE = os.environ.get("LLM_SERVABLE", "/data/llm/models/servable")
RUN = "/data/llm/run"
ACTIVITY = os.path.join(RUN, "model-activity.json")
KEEP = int(os.environ.get("LLM_CACHE_KEEP", "2"))
TTL = int(os.environ.get("LLM_CACHE_TTL", str(2 * 3600)))
# Throttle for the background copy. NFS is shared by every node, and the copy
# only ever runs while nothing is loading, but a batch job may still be reading.
COPY_MBPS = float(os.environ.get("LLM_CACHE_MBPS", "150"))
# Never fill the cache filesystem past this, whatever the keep set says.
MIN_FREE_FRAC = 0.10
POLL = 10

NODE = socket.gethostname().split(".")[0]
FARM = os.path.join(CACHE, "servable")
MODELS = os.path.join(CACHE, "models")
RPC = os.path.join(CACHE, "rpc")
STATE = os.path.join(CACHE, "state.json")
STAGED = os.path.join(RUN, "staged.json")                 # written by slerm-orch
STAGE_COPIES = os.path.join(CACHE, "staged-copies.json")  # local copies the stager made
STAGE_FILES = os.path.join(CACHE, "staged-files.json")    # rpc file -> staged models
STAGE_LOCK = os.path.join(CACHE, "stage.lock")
HASHES = os.path.join(RUN, "tensor-hashes")               # shared: tensor -> fnv hash
RPC_STAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rpc-stage")


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), "model-cache:", *a, flush=True)


def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    tmp = "%s.%s.%d.tmp" % (path, NODE, os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def fs_free(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize, st.f_blocks * st.f_frsize


def relink(link, target):
    """Atomically point link at target (rename over the old link)."""
    tmp = link + ".new"
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
    os.symlink(target, tmp)
    os.replace(tmp, link)


def usable():
    return os.path.isdir(CACHE) and os.access(CACHE, os.W_OK)


def keep_set(last_used, now):
    live = [(t, m) for m, t in last_used.items() if now - t < TTL]
    return set(m for _, m in sorted(live, reverse=True)[:KEEP]) | staged_ids()


def staged_list():
    return (read_json(STAGED, {}) or {}).get("models") or []


def staged_ids():
    return set(e["model"] for e in staged_list())


# ------------------------------------------------------------------ router --

def servable_models():
    """{model id: real NFS path} for everything the router offers. With
    LLM_ALLOWED (written by slerm_topo.py) only the models that can run on the
    current device set: the router lists every file in its models dir, and an
    entry it cannot load sits in the picker and fails only when chosen."""
    allowed = None
    if os.environ.get("LLM_ALLOWED"):
        try:
            allowed = set(x.strip() for x in open(os.environ["LLM_ALLOWED"]) if x.strip())
        except OSError:
            allowed = None
    out = {}
    for name in sorted(os.listdir(SERVABLE)):
        if not name.endswith(".gguf"):
            continue
        if allowed is not None and name[:-5] not in allowed:
            continue
        real = os.path.realpath(os.path.join(SERVABLE, name))
        if os.path.isfile(real):
            out[name[:-5]] = real
    return out


def local_copy(real):
    return os.path.join(MODELS, os.path.basename(real))


def copy_is_current(real, meta):
    """A local copy counts only if it matches the NFS file it was made from."""
    loc = local_copy(real)
    if not os.path.isfile(loc) or not meta:
        return False
    try:
        st = os.stat(real)
    except OSError:
        return False
    return (os.path.getsize(loc) == st.st_size == meta.get("size")
            and int(st.st_mtime) == meta.get("src_mtime"))


def init_farm(state=None):
    os.makedirs(FARM, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    state = state if state is not None else read_json(STATE, {})
    cached = state.setdefault("cached", {})
    models = servable_models()
    for mid, meta in (read_json(STAGE_COPIES, {}) or {}).items():
        if mid not in cached and mid in models and copy_is_current(models[mid], meta):
            cached[mid] = meta
    for mid, real in models.items():
        link = os.path.join(FARM, mid + ".gguf")
        want = local_copy(real) if copy_is_current(real, cached.get(mid)) else real
        if os.path.realpath(link) != os.path.realpath(want):
            relink(link, want)
    for name in os.listdir(FARM):
        if name.endswith(".gguf") and name[:-5] not in models:
            os.unlink(os.path.join(FARM, name))
    return models


class Copier(threading.Thread):
    """Throttled NFS -> local copy that pauses while any model is loading."""

    def __init__(self, mid, real, busy):
        super().__init__(daemon=True, name="copy-" + mid)
        self.mid, self.real, self.busy = mid, real, busy
        self.ok = False
        self.cancel = False

    def run(self):
        dst = local_copy(self.real)
        part = dst + ".part"
        chunk = 8 << 20
        t0, done = time.time(), 0
        try:
            with open(self.real, "rb") as src, open(part, "wb") as out:
                while not self.cancel:
                    while self.busy.is_set() and not self.cancel:
                        time.sleep(2)
                    buf = src.read(chunk)
                    if not buf:
                        break
                    out.write(buf)
                    done += len(buf)
                    ahead = done / (COPY_MBPS * 1e6) - (time.time() - t0)
                    if ahead > 0:
                        time.sleep(ahead)
                    # Do not let the copy push the resident model out of RAM.
                    try:
                        os.posix_fadvise(src.fileno(), 0, done, os.POSIX_FADV_DONTNEED)
                    except (AttributeError, OSError):
                        pass
                out.flush()
                os.fsync(out.fileno())
            if self.cancel:
                os.unlink(part)
                return
            if os.path.getsize(part) != os.path.getsize(self.real):
                log("copy of", self.mid, "came out the wrong size; discarded")
                os.unlink(part)
                return
            os.replace(part, dst)
            self.ok = True
            log("cached %s (%.1f GB, %.0f MB/s)" % (self.mid, done / 1e9, done / 1e6 / max(1, time.time() - t0)))
        except OSError as e:
            log("copy of", self.mid, "failed:", e)
            try:
                os.unlink(part)
            except OSError:
                pass


def router_status(port):
    with urllib.request.urlopen("http://127.0.0.1:%d/models" % port, timeout=10) as r:
        data = json.load(r).get("data", [])
    return {m["id"]: (m.get("status") or {}).get("value", "") for m in data}


def evict(mid, real, state):
    link = os.path.join(FARM, mid + ".gguf")
    if os.path.lexists(link):
        relink(link, real)          # NFS first, so no load can open a vanishing file
    try:
        os.unlink(local_copy(real))
    except FileNotFoundError:
        pass
    state["cached"].pop(mid, None)
    log("evicted", mid)


def run_router(port, job):
    if not usable():
        log("no writable", CACHE, "- cache disabled, router reads NFS")
        return
    state = read_json(STATE, {})
    state.setdefault("cached", {})
    last_used = state.setdefault("last_used", {})
    for f in os.listdir(MODELS) if os.path.isdir(MODELS) else []:
        if f.endswith(".part"):                 # a copy killed mid-way
            os.unlink(os.path.join(MODELS, f))
    threading.Thread(target=stager_loop, daemon=True, name="stager").start()
    loads = []          # [{"model", "start", "end"}], newest last
    loading = {}        # model -> start
    busy = threading.Event()
    copier = None
    while True:
        now = time.time()
        try:
            models = init_farm(state)
            status = router_status(port)
        except (OSError, ValueError) as e:
            log("router not answering:", e)
            time.sleep(POLL)
            continue

        for mid, st in status.items():
            if st == "loading" and mid not in loading:
                loading[mid] = now
                loads.append({"model": mid, "start": now, "end": None})
            elif st != "loading" and mid in loading:
                start = loading.pop(mid)
                for rec in loads:
                    if rec["model"] == mid and rec["start"] == start:
                        rec["end"] = now
            if st in ("loaded", "loading"):
                last_used[mid] = now
        loads = [r for r in loads if (r["end"] or now) > now - TTL][-50:]
        busy.set() if loading else busy.clear()
        keep = keep_set(last_used, now)

        for mid in list(state["cached"]):
            real = models.get(mid)
            if real is None or mid not in keep or not copy_is_current(real, state["cached"][mid]):
                if copier and copier.mid == mid:
                    copier.cancel = True
                if real is None:
                    state["cached"].pop(mid, None)
                else:
                    evict(mid, real, state)

        if copier and copier.is_alive() and copier.mid not in keep:
            copier.cancel = True
        if copier and not copier.is_alive():
            if copier.ok:
                st = os.stat(copier.real)
                state["cached"][copier.mid] = {"size": st.st_size, "src_mtime": int(st.st_mtime),
                                               "cached_at": int(now)}
                init_farm(state)
            copier = None
        if copier is None:
            for mid in sorted(keep, key=lambda m: -last_used.get(m, 0)):
                real = models.get(mid)
                if (real and mid not in state["cached"] and status.get(mid) == "loaded"
                        and not loading):
                    free, total = fs_free(CACHE)
                    need = os.path.getsize(real)
                    if free - need < total * MIN_FREE_FRAC:
                        log("no room to cache %s (%.1f GB free)" % (mid, free / 1e9))
                        continue
                    copier = Copier(mid, real, busy)
                    copier.start()
                    log("caching", mid)
                    break

        state["last_used"] = {m: t for m, t in last_used.items() if now - t < 7 * 86400}
        last_used = state["last_used"]
        write_json(STATE, state)
        publish = {"node": NODE, "role": "router", "job": job, "updated": int(now),
                   "keep": sorted(keep), "ttl": TTL, "keep_n": KEEP,
                   "last_used": {m: int(t) for m, t in last_used.items() if now - t < TTL},
                   "loads": loads,
                   "cached": {m: {"bytes": v["size"], "cached_at": v.get("cached_at")}
                              for m, v in state["cached"].items()},
                   "copying": copier.mid if copier else None,
                   "free_bytes": fs_free(CACHE)[0]}
        try:
            write_json(ACTIVITY, publish)
            write_json(os.path.join(RUN, "cache-%s.json" % NODE), publish)
        except OSError as e:
            log("could not publish:", e)
        time.sleep(POLL)


# ------------------------------------------------------------------ worker --

def run_worker(job):
    if not usable():
        log("no writable", CACHE, "- rpc cache disabled")
        return
    os.makedirs(RPC, exist_ok=True)
    manifest_path = os.path.join(CACHE, "rpc-manifest.json")
    manifest = read_json(manifest_path, {})     # file -> {"models": [...], "mtime": t}
    threading.Thread(target=stager_loop, daemon=True, name="stager").start()
    while True:
        now = time.time()
        act = read_json(ACTIVITY, {})
        loads = act.get("loads", [])
        last_used = act.get("last_used", {})
        keep = keep_set({m: float(t) for m, t in last_used.items()}, now)
        present = {}
        for name in os.listdir(RPC):
            p = os.path.join(RPC, name)
            try:
                st = os.stat(p)
            except FileNotFoundError:
                continue
            present[name] = st
        staged_files = read_json(STAGE_FILES, {}) or {}
        for name, st in present.items():
            if name in manifest:
                continue
            # A file the stager wrote belongs to its staged model only; the
            # load-window guess would pin it on whatever loaded a minute ago.
            owners = [] if name in staged_files else \
                sorted(set(r["model"] for r in loads
                           if r["start"] - 30 <= st.st_mtime <= (r["end"] or now) + 60))
            manifest[name] = {"models": owners, "mtime": st.st_mtime, "bytes": st.st_size}
        for name in list(manifest):
            if name not in present:
                del manifest[name]

        # Evict files whose models all fell out of the keep set, and anything
        # unattributed past the TTL. The 10-minute floor keeps a load that is
        # still being written (and not yet in a fresh activity file) safe.
        for name, meta in list(manifest.items()):
            age = now - meta["mtime"]
            owners = meta["models"] + staged_files.get(name, [])
            stale = (not any(m in keep for m in owners)) if owners else age > TTL
            if stale and age > 600:
                try:
                    os.unlink(os.path.join(RPC, name))
                except FileNotFoundError:
                    pass
                del manifest[name]
        free, total = fs_free(CACHE)
        if free < total * MIN_FREE_FRAC:
            for name, meta in sorted(manifest.items(), key=lambda kv: kv[1]["mtime"]):
                if free >= total * MIN_FREE_FRAC:
                    break
                try:
                    os.unlink(os.path.join(RPC, name))
                    free += meta["bytes"]
                except FileNotFoundError:
                    pass
                del manifest[name]
        write_json(manifest_path, manifest)
        per = {}
        for meta in manifest.values():
            for m in meta["models"] or ["(unattributed)"]:
                per[m] = per.get(m, 0) + meta["bytes"]
        try:
            write_json(os.path.join(RUN, "cache-%s.json" % NODE),
                       {"node": NODE, "role": "worker", "job": job, "updated": int(now),
                        "keep": sorted(keep), "rpc_bytes": per, "free_bytes": free})
        except OSError as e:
            log("could not publish:", e)
        time.sleep(POLL * 3)


# ------------------------------------------------------------------- stage --

def my_work(staged):
    """[(entry, role, dev_index)] this node owes the staged models."""
    out = []
    for e in staged:
        devs = e.get("devices") or []
        if NODE not in devs:
            continue
        i = devs.index(NODE)
        if i == len(devs) - 1:
            out.append((e, "router", i))
        elif i < len(e.get("split") or []) and float(e["split"][i]) > 0:
            out.append((e, "worker", i))
    return out


def loading_now():
    """A model load is running somewhere (the router's activity file)."""
    act = read_json(ACTIVITY, {})
    if time.time() - (act.get("updated") or 0) > 60:
        return False
    return any(r.get("end") is None for r in act.get("loads") or [])


def lru_keep():
    """The router's own keep set (last models used), without the staged ones."""
    now = time.time()
    lu = (read_json(ACTIVITY, {}) or {}).get("last_used") or {}
    live = [(float(t), m) for m, t in lu.items() if now - float(t) < TTL]
    return set(m for _, m in sorted(live, reverse=True)[:KEEP])


class Stager:
    def __init__(self):
        self.status = {}        # model -> {"role", "state", "done", "total", "at", ...}
        self.last_pub = 0

    def publish(self, active, force=False):
        now = time.time()
        if not force and now - self.last_pub < 5:
            return
        self.last_pub = now
        try:
            write_json(os.path.join(RUN, "stage-%s.json" % NODE),
                       {"node": NODE, "updated": int(now), "active": active,
                        "free_bytes": fs_free(CACHE)[0], "models": self.status})
        except OSError as e:
            log("stage: could not publish:", e)

    def set(self, mid, **kw):
        self.status.setdefault(mid, {}).update(kw)
        self.publish(True)

    def wait_idle(self, mid):
        paused = False
        while loading_now():
            if not paused:
                self.set(mid, state="paused", note="a model is loading")
                paused = True
            time.sleep(5)
        if paused:
            self.set(mid, state="staging", note="")

    # -- router node: a full local copy, picked up by the link farm ------------

    def stage_router(self, e):
        mid, real = e["model"], e["path"]
        st = os.stat(real)
        meta = {"size": st.st_size, "src_mtime": int(st.st_mtime)}
        copies = read_json(STAGE_COPIES, {}) or {}
        dst = local_copy(real)
        self.set(mid, role="router", total=st.st_size, done=0)
        if copy_is_current(real, copies.get(mid)) or copy_is_current(
                real, (read_json(STATE, {}).get("cached") or {}).get(mid)):
            copies[mid] = dict(meta, cached_at=copies.get(mid, {}).get("cached_at", int(time.time())))
            write_json(STAGE_COPIES, copies)
            return self.set(mid, state="done", done=st.st_size)
        os.makedirs(MODELS, exist_ok=True)
        free, total = fs_free(CACHE)
        if free - st.st_size < total * MIN_FREE_FRAC:
            self.make_room_router(st.st_size)
            free, total = fs_free(CACHE)
            if free - st.st_size < total * MIN_FREE_FRAC:
                return self.set(mid, state="no room",
                                note="%.1f GB free, needs %.1f GB" % (free / 1e9, st.st_size / 1e9))
        self.set(mid, state="staging")
        part = dst + ".stage.part"
        t0, done = time.time(), 0
        with open(real, "rb") as src, open(part, "wb") as out:
            while True:
                if mid not in staged_ids():
                    out.close()
                    os.unlink(part)
                    return self.status.pop(mid, None)
                self.wait_idle(mid)
                buf = src.read(8 << 20)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                ahead = done / (COPY_MBPS * 1e6) - (time.time() - t0)
                if ahead > 0:
                    time.sleep(ahead)
                try:
                    os.posix_fadvise(src.fileno(), 0, done, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
                self.status[mid]["done"] = done
                self.publish(True)
            out.flush()
            os.fsync(out.fileno())
        if os.path.getsize(part) != st.st_size:
            os.unlink(part)
            return self.set(mid, state="error", note="copy came out the wrong size")
        os.replace(part, dst)
        copies = read_json(STAGE_COPIES, {}) or {}
        copies[mid] = dict(meta, cached_at=int(time.time()))
        write_json(STAGE_COPIES, copies)
        link = os.path.join(FARM, mid + ".gguf")
        if os.path.lexists(link):
            relink(link, dst)       # takes effect on the router's next load of it
        log("staged %s (%.1f GB, %.0f MB/s)" % (mid, done / 1e9, done / 1e6 / max(1, time.time() - t0)))
        self.set(mid, state="done", done=done)

    def make_room_router(self, need):
        """Drop local copies that are neither staged nor among the last used."""
        keep = staged_ids() | lru_keep()
        state = read_json(STATE, {})
        cached = dict(state.get("cached") or {})
        cached.update(read_json(STAGE_COPIES, {}) or {})
        for mid in list(cached):
            if mid in keep:
                continue
            real = servable_models_all().get(mid)
            if real:
                self.evict_copy(mid, real)
            free, total = fs_free(CACHE)
            if free - need >= total * MIN_FREE_FRAC:
                return

    def evict_copy(self, mid, real):
        link = os.path.join(FARM, mid + ".gguf")
        if os.path.lexists(link):
            relink(link, real)
        try:
            os.unlink(local_copy(real))
            log("stage: dropped local copy of", mid)
        except FileNotFoundError:
            pass
        copies = read_json(STAGE_COPIES, {}) or {}
        if copies.pop(mid, None) is not None:
            write_json(STAGE_COPIES, copies)

    # -- worker: the rpc cache files it would be sent --------------------------

    def stage_worker(self, e, idx):
        mid, real = e["model"], e["path"]
        split = [float(x) for x in e["split"]]
        tensors = model_stage.worker_tensors(real, split, idx)
        st = os.stat(real)
        hpath = os.path.join(HASHES, "%s-%d-%d.json" % (os.path.basename(real), st.st_size, int(st.st_mtime)))
        hashes = read_json(hpath, {}) or {}
        total = sum(t[2] for t in tensors)
        have = [t for t in tensors if t[0] in hashes and os.path.isfile(os.path.join(RPC, hashes[t[0]]))]
        need = [t for t in tensors if t not in have]
        done = sum(t[2] for t in have)
        self.set(mid, role="worker", dev_index=idx, total=total, done=done,
                 tensors=len(tensors))
        os.makedirs(RPC, exist_ok=True)
        files = read_json(STAGE_FILES, {}) or {}

        def own(h):
            owners = files.setdefault(h, [])
            if mid not in owners:
                owners.append(mid)

        for t in have:
            own(hashes[t[0]])
        write_json(STAGE_FILES, files)
        if need:
            free, ftotal = fs_free(CACHE)
            nbytes = sum(t[2] for t in need)
            if free - nbytes < ftotal * MIN_FREE_FRAC:
                return self.set(mid, state="no room",
                                note="%.1f GB free, needs %.1f GB" % (free / 1e9, nbytes / 1e9))
            self.set(mid, state="staging")
            proc = subprocess.Popen([RPC_STAGE, real, RPC, str(COPY_MBPS)], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, universal_newlines=True, bufsize=1)
            try:
                for name, off, size in need:
                    if mid not in staged_ids():
                        return self.status.pop(mid, None)
                    self.wait_idle(mid)
                    proc.stdin.write("%d %d\n" % (off, size))
                    proc.stdin.flush()
                    line = proc.stdout.readline().split()
                    if len(line) != 2:
                        return self.set(mid, state="error", note="rpc-stage failed on " + name)
                    hashes[name] = line[1]
                    own(line[1])
                    write_json(STAGE_FILES, files)      # before the worker loop can see it
                    done += size
                    self.status[mid]["done"] = done
                    self.publish(True)
            finally:
                proc.stdin.close()
                proc.wait()
            try:
                os.makedirs(HASHES, exist_ok=True)
                merged = read_json(hpath, {}) or {}
                merged.update(hashes)
                write_json(hpath, merged)
            except OSError as e2:
                log("stage: could not save tensor hashes:", e2)
        if need:
            log("staged %s: %d rpc tensors (%.1f GB)" % (mid, len(need), sum(t[2] for t in need) / 1e9))
        self.set(mid, state="done", done=total)

    # -- unstage ---------------------------------------------------------------

    def cleanup(self, staged):
        """Remove what the stager made for models no longer staged. Anything the
        router or worker loops keep for their own reasons (last used) stays."""
        ids = set(e["model"] for e in staged)
        keep = ids | lru_keep()
        models = servable_models_all()
        router_cached = read_json(STATE, {}).get("cached") or {}
        for mid in list((read_json(STAGE_COPIES, {}) or {}).keys()):
            if mid in ids:
                continue
            # Still wanted by the router's own policy AND known to it (a router
            # that ran since merged it): hand it over. Otherwise remove it.
            if mid not in models or (mid in keep and mid in router_cached):
                copies = read_json(STAGE_COPIES, {}) or {}
                copies.pop(mid, None)
                write_json(STAGE_COPIES, copies)
            else:
                self.evict_copy(mid, models[mid])
        files = read_json(STAGE_FILES, {}) or {}
        manifest = read_json(os.path.join(CACHE, "rpc-manifest.json"), {}) or {}
        changed = False
        for h, owners in list(files.items()):
            live = [m for m in owners if m in ids]
            if live == owners:
                continue
            changed = True
            if live:
                files[h] = live
                continue
            del files[h]
            if any(m in keep for m in (manifest.get(h) or {}).get("models") or []):
                continue
            try:
                os.unlink(os.path.join(RPC, h))
            except FileNotFoundError:
                pass
        if changed:
            write_json(STAGE_FILES, files)
        for mid in list(self.status):
            if mid not in ids:
                del self.status[mid]

    def run_pass(self):
        """One pass over this node's staged work. False if another stager owns it."""
        lockf = open(STAGE_LOCK, "a")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lockf.close()
            return False
        try:
            staged = staged_list()
            self.cleanup(staged)
            for e, role, idx in my_work(staged):
                at = self.status.get(e["model"], {}).get("at")
                if at == e.get("at") and self.status[e["model"]].get("state") == "done":
                    continue
                self.status[e["model"]] = {"at": e.get("at"), "state": "queued", "role": role}
            self.publish(True, force=True)
            for e, role, idx in my_work(staged):
                s = self.status.get(e["model"], {})
                if s.get("state") == "done":
                    continue
                try:
                    if role == "router":
                        self.stage_router(e)
                    else:
                        self.stage_worker(e, idx)
                except Exception as ex:                          # noqa: BLE001
                    log("stage", e["model"], "failed:", ex)
                    self.set(e["model"], state="error", note=str(ex)[:200])
            self.publish(False, force=True)
            return True
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)
            lockf.close()

    def pending(self):
        work = my_work(staged_list())
        return any(self.status.get(e["model"], {}).get("state") not in ("done", "no room", "error")
                   or self.status.get(e["model"], {}).get("at") != e.get("at") for e, _, _ in work)


def servable_models_all():
    """{model id: real NFS path} for every servable model, ignoring LLM_ALLOWED."""
    out = {}
    for name in os.listdir(SERVABLE):
        if name.endswith(".gguf"):
            real = os.path.realpath(os.path.join(SERVABLE, name))
            if os.path.isfile(real):
                out[name[:-5]] = real
    return out


def stager_loop():
    """Thread of the router/worker loops: stage whatever is owed, every 30 s."""
    if not usable() or not os.access(RPC_STAGE, os.X_OK):
        return
    s = Stager()
    while True:
        try:
            s.run_pass()
        except Exception as e:                                   # noqa: BLE001
            log("stager:", e)
        time.sleep(30)


def run_stage(max_s=6 * 3600):
    """The llm-stage job: stage until done (or pushed out), then exit."""
    if not usable():
        log("stage: no writable", CACHE)
        return
    s = Stager()
    t0 = time.time()
    while time.time() - t0 < max_s:
        if not s.run_pass():
            log("stage: another stager holds", STAGE_LOCK, "- leaving it to that one")
            return
        if not s.pending():
            log("stage: nothing left to stage on", NODE)
            return
        time.sleep(15)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    job = os.environ.get("SLURM_JOB_ID", "")
    if mode == "init-farm":
        if not usable():
            print(SERVABLE)             # no cache here: the router uses NFS as before
            return
        init_farm()
        print(FARM)
    elif mode == "router":
        run_router(int(sys.argv[2]), job)
    elif mode == "worker":
        run_worker(job)
    elif mode == "stage":
        run_stage()
    else:
        sys.exit("usage: model-cache.py init-farm | router PORT | worker | stage")


if __name__ == "__main__":
    main()
