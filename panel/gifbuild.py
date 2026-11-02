"""Turn a project's rendered PNGs into one animated GIF.

Pillow only -- nothing extra gets installed on the head node. Two things keep
this workable at full resolution over a couple of hundred frames:

  * a single master palette shared by every frame, so Pillow's GIF writer can
    delta-encode each frame against the last (radar plots share a colormap, so
    one 256-colour table is faithful and compresses far better than a
    per-frame adaptive palette);
  * frames are opened, composited and quantised one at a time and handed to
    the encoder through a generator, so peak memory is a few frames rather
    than the whole stack of RGBA bitmaps.
"""
import hashlib
import json
import os
import tempfile
import threading
import time

from PIL import Image

DEFAULTS = {"width": 0, "stride": 1, "fps": 8}   # width 0 == native resolution
DEFAULT_GROUP = "all"
PALETTE_SAMPLES = 12

_builds = {}
_lock = threading.Lock()


def normalise(params):
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        if params.get(key) not in (None, ""):
            try:
                out[key] = int(params[key])
            except (TypeError, ValueError):
                pass
    out["width"] = max(0, min(out["width"], 4000))
    out["stride"] = max(1, min(out["stride"], 100))
    out["fps"] = max(1, min(out["fps"], 50))
    group = params.get("group") or DEFAULT_GROUP
    out["group"] = str(group)[:64]
    return out


def select_frames(product, params):
    frames = product.frames_in_group(params.get("group"))
    return frames[::params["stride"]]


def cache_key(product, params, frames):
    digest = hashlib.sha1()
    digest.update(json.dumps(params, sort_keys=True).encode())
    for f in frames:
        digest.update(("%s:%s:%s|" % (f["name"], f["mtime"], f["size"])).encode())
    return digest.hexdigest()[:16]


def cache_path(product, key):
    return os.path.join(product.cache_dir, "%s-%s.gif" % (product.id, key))


def describe(product, params):
    """What the current cache state is, without building anything."""
    params = normalise(params)
    frames = select_frames(product, params)
    key = cache_key(product, params, frames)
    path = cache_path(product, key)
    ready = os.path.isfile(path)
    with _lock:
        build = dict(_builds.get(key) or {})
    return {
        "key": key,
        "params": params,
        "frame_count": len(frames),
        "group_frames": len(product.frames_in_group(params.get("group"))),
        "total_frames": len(product.frames()),
        "groups": product.groups(),
        "ready": ready,
        "size": os.path.getsize(path) if ready else None,
        "built_at": int(os.path.getmtime(path)) if ready else None,
        "build": build or None,
    }


def _open_rgb(path):
    """Load a frame as RGB, flattening any alpha onto white."""
    im = Image.open(path)
    im.load()
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba).convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    return im


def _resized(im, width):
    if not width or width >= im.width:
        return im
    height = max(1, round(im.height * width / im.width))
    return im.resize((width, height), Image.LANCZOS)


def _master_palette(paths, width):
    """One 256-colour table derived from a sample spread across the run.

    The samples are quantised at the *output* resolution, never smaller.
    Downscaling first blends thin features -- map borders, coastlines, grid
    lines -- into the background, so the palette ends up with no true black
    and those lines come out tinted (they went dark red in an earlier build).
    """
    if len(paths) > PALETTE_SAMPLES:
        step = len(paths) / float(PALETTE_SAMPLES)
        sample = [paths[int(i * step)] for i in range(PALETTE_SAMPLES)]
    else:
        sample = list(paths)

    tiles = []
    for path in sample:
        try:
            im = _open_rgb(path)
        except OSError:
            continue
        tiles.append(_resized(im, width))
    if not tiles:
        raise RuntimeError("no readable frames to build a palette from")

    strip_w = max(t.width for t in tiles)
    strip_h = sum(t.height for t in tiles)
    strip = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
    y = 0
    for t in tiles:
        strip.paste(t, (0, y))
        y += t.height
    return strip.quantize(colors=256, method=Image.MEDIANCUT)


def build(product, params, force=False):
    """Start (or join) a build. Returns the cache key immediately."""
    params = normalise(params)
    frames = select_frames(product, params)
    if not frames:
        raise RuntimeError("this product has no rendered frames yet")
    key = cache_key(product, params, frames)
    path = cache_path(product, key)

    with _lock:
        state = _builds.get(key)
        if state and state.get("state") == "running":
            return key
        if os.path.isfile(path) and not force:
            return key
        _builds[key] = {"state": "running", "done": 0, "total": len(frames),
                        "started": time.time(), "error": None, "path": path}

    thread = threading.Thread(target=_worker, args=(product, params, frames, key, path),
                             name="gif-%s" % key, daemon=True)
    thread.start()
    return key


def _worker(product, params, frames, key, path):
    def mark(**fields):
        with _lock:
            _builds.setdefault(key, {}).update(fields)

    tmp = None
    try:
        os.makedirs(product.cache_dir, exist_ok=True)
        paths = [os.path.join(product.out_dir, f["name"]) for f in frames]
        width = params["width"]
        duration = int(round(1000.0 / params["fps"]))

        mark(stage="palette")
        master = _master_palette(paths, width)

        first = _resized(_open_rgb(paths[0]), width).quantize(
            palette=master, dither=Image.Dither.NONE)
        mark(stage="encoding", done=1)

        def rest():
            for index, frame_path in enumerate(paths[1:], start=2):
                try:
                    im = _resized(_open_rgb(frame_path), width).quantize(
                        palette=master, dither=Image.Dither.NONE)
                except OSError:
                    continue
                mark(done=index)
                yield im

        fd, tmp = tempfile.mkstemp(dir=product.cache_dir, suffix=".gif.part")
        os.close(fd)
        first.save(tmp, format="GIF", save_all=True, append_images=rest(),
                   duration=duration, loop=0, disposal=2, optimize=False)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
        tmp = None
        mark(state="done", stage="done", done=len(frames),
             finished=time.time(), size=os.path.getsize(path))
    except Exception as exc:  # noqa: BLE001 - report the failure in the UI
        mark(state="error", error="%s: %s" % (type(exc).__name__, exc),
             finished=time.time())
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def prune(product, keep=4):
    """Drop stale cached GIFs, newest `keep` survive."""
    try:
        entries = [os.path.join(product.cache_dir, n)
                   for n in os.listdir(product.cache_dir)
                   if n.startswith(product.id + "-") and n.endswith(".gif")]
    except OSError:
        return 0
    entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    removed = 0
    for path in entries[keep:]:
        try:
            os.unlink(path)
            removed += 1
        except OSError:
            pass
    return removed
