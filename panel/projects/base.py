"""Shared shape for every project the panel can drive.

A project owns the job-submission side (how work is listed and handed to
Slurm). A *product* is one derived image series that comes out of it --
reflectivity, rotation, storm tracks -- each with its own output directory,
frame index, dataset grouping and GIF cache. Splitting them is what lets one
tab show several views of the same scans without them fighting over state.
"""
import glob
import os
import subprocess
from dataclasses import dataclass, field


class ProjectError(RuntimeError):
    pass


@dataclass
class Param:
    """One input on a project's run form.

    `kind` picks the widget: "text", or "day" for a date picker paired with a
    UTC/Central selector, annotated with the UTC window it resolves to.
    """
    name: str
    label: str
    default: str = ""
    help: str = ""
    placeholder: str = ""
    kind: str = "text"

    def as_dict(self):
        return {"name": self.name, "label": self.label, "default": self.default,
                "help": self.help, "placeholder": self.placeholder,
                "kind": self.kind}


@dataclass
class Product:
    """One renderable series belonging to a project."""
    id: str
    label: str
    description: str
    out_dir: str
    cache_dir: str
    #: helper script run once per work item, or None for products built by the
    #: project as a whole (a per-day summary plot, say)
    script: str = None
    #: "series" animates over time; "single" is one standing image per day
    kind: str = "series"
    frame_glob: str = "*.png"
    #: callable(frame_name) -> (group_key, group_label) or None
    group_of: object = None
    #: callable(frame_name) -> UTC epoch seconds, or None
    time_of: object = None
    #: shown layered under another product when comparing
    overlays: str = None
    #: per-task memory, when this product needs more than the project default
    default_mem: str = None
    #: callable(project, product, params) -> (list_path, count); overrides the
    #: project's own lister for products whose work items are not radar scans
    make_list: object = None
    #: extra note shown on the product pane
    note: str = None
    #: callable(project, product, group) -> output path; a second pass that
    #: turns per-item sidecars into a single figure
    assemble: object = None

    def as_dict(self):
        return {"id": self.id, "label": self.label,
                "description": self.description, "kind": self.kind,
                "out_dir": self.out_dir, "groupable": self.group_of is not None,
                "overlays": self.overlays, "default_mem": self.default_mem,
                "note": self.note, "assembles": self.assemble is not None}

    def ensure_dirs(self):
        for d in (self.out_dir, self.cache_dir):
            os.makedirs(d, exist_ok=True)

    def frames(self):
        """Rendered images, oldest name first (names sort chronologically)."""
        out = []
        for path in sorted(glob.glob(os.path.join(self.out_dir, self.frame_glob))):
            try:
                st = os.stat(path)
            except OSError:
                continue
            name = os.path.basename(path)
            out.append({"name": name, "size": st.st_size,
                        "mtime": int(st.st_mtime),
                        "t": self.time_of(name) if self.time_of else None})
        return out

    def groups(self):
        """Bucket frames into datasets (for NEXRAD: one site + UTC day)."""
        if not self.group_of:
            return []
        buckets = {}
        for frame in self.frames():
            found = self.group_of(frame["name"])
            if not found:
                continue
            key, label = found
            bucket = buckets.setdefault(key, {
                "key": key, "label": label, "count": 0,
                "first": frame["name"], "last": frame["name"],
                "bytes": 0, "mtime": frame["mtime"],
                "first_t": frame["t"], "last_t": frame["t"],
            })
            bucket["count"] += 1
            bucket["last"] = frame["name"]
            bucket["bytes"] += frame["size"]
            bucket["mtime"] = max(bucket["mtime"], frame["mtime"])
            if frame["t"] is not None:
                if bucket["first_t"] is None or frame["t"] < bucket["first_t"]:
                    bucket["first_t"] = frame["t"]
                if bucket["last_t"] is None or frame["t"] > bucket["last_t"]:
                    bucket["last_t"] = frame["t"]
        return sorted(buckets.values(), key=lambda b: b["key"])

    def frames_in_group(self, group=None):
        frames = self.frames()
        if not group or group == "all" or not self.group_of:
            return frames
        out = []
        for frame in frames:
            found = self.group_of(frame["name"])
            if found and found[0] == group:
                out.append(frame)
        return out

    def frame_path(self, name):
        """Resolve a frame name safely inside out_dir."""
        if "/" in name or name.startswith("."):
            raise ProjectError("bad frame name")
        real = os.path.realpath(os.path.join(self.out_dir, name))
        if not real.startswith(os.path.realpath(self.out_dir) + os.sep):
            raise ProjectError("frame outside output directory")
        if not os.path.isfile(real):
            raise ProjectError("no such frame")
        return real

    def status(self):
        frames = self.frames()
        return {
            "id": self.id,
            "frame_count": len(frames),
            "frame_bytes": sum(f["size"] for f in frames),
            "newest_frame": frames[-1]["name"] if frames else None,
            "newest_frame_mtime": frames[-1]["mtime"] if frames else None,
            "groups": self.groups(),
        }


@dataclass
class Project:
    id: str
    label: str
    description: str
    python: str
    scripts_dir: str
    data_dir: str
    lists_dir: str
    out_dir: str
    logs_dir: str
    cache_dir: str
    array_script: str
    job_name: str
    view: str = "frames"
    products: list = field(default_factory=list)
    default_partition: str = "compute"
    default_timelimit: str = "00:10:00"
    default_throttle: int = 32
    default_mem: str = "2G"
    params: list = field(default_factory=list)

    # --- hooks a project module overrides -----------------------------------
    #: callable(project, params) -> (list_path, item_count)
    make_list: object = None
    #: callable(project, list_path, count, options) -> [sbatch args]
    sbatch_args: object = None
    #: callable(date_str, mode) -> dict describing the UTC window a day covers
    day_info: object = None
    #: callable(project, run_id, job_id, list_path, options), invoked after the
    #: primary Slurm job has been submitted successfully
    post_submit: object = None

    def as_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "view": self.view,
            "params": [p.as_dict() for p in self.params],
            "products": [p.as_dict() for p in self.products],
            "default_partition": self.default_partition,
            "default_timelimit": self.default_timelimit,
            "default_throttle": self.default_throttle,
            "default_mem": self.default_mem,
            "out_dir": self.out_dir,
            "logs_dir": self.logs_dir,
            "lists_dir": self.lists_dir,
            "job_name": self.job_name,
        }

    # --- products -----------------------------------------------------------
    def product(self, product_id=None):
        if not self.products:
            raise ProjectError("project %r has no products" % self.id)
        if not product_id:
            return self.products[0]
        for product in self.products:
            if product.id == product_id:
                return product
        raise ProjectError("unknown product %r" % product_id)

    def ensure_dirs(self):
        for d in (self.lists_dir, self.out_dir, self.logs_dir, self.cache_dir):
            os.makedirs(d, exist_ok=True)
        for product in self.products:
            product.ensure_dirs()

    # --- convenience delegating to the default product ----------------------
    def frames(self):
        return self.product().frames()

    def status(self):
        lists = []
        for path in sorted(glob.glob(os.path.join(self.lists_dir, "*.txt"))):
            try:
                with open(path) as fh:
                    count = sum(1 for _ in fh)
            except OSError:
                continue
            lists.append({"name": os.path.basename(path), "count": count,
                          "mtime": int(os.path.getmtime(path))})
        logs = glob.glob(os.path.join(self.logs_dir, "*.out"))
        products = {p.id: p.status() for p in self.products}
        default = self.products[0].id if self.products else None
        return {
            "lists": lists,
            "list_total": sum(l["count"] for l in lists),
            "log_count": len(logs),
            "products": products,
            # kept flat for the status tiles
            "frame_count": products[default]["frame_count"] if default else 0,
            "frame_bytes": products[default]["frame_bytes"] if default else 0,
            "newest_frame": products[default]["newest_frame"] if default else None,
            "newest_frame_mtime": (products[default]["newest_frame_mtime"]
                                   if default else None),
            "groups": products[default]["groups"] if default else [],
        }


def run_helper(argv, timeout=900):
    """Run a project helper script and return its stdout."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                           env={"PATH": "/usr/bin:/bin", "HOME": "/home/slerm"})
    except subprocess.TimeoutExpired:
        raise ProjectError("%s timed out after %ss" % (argv[0], timeout))
    except FileNotFoundError:
        raise ProjectError("%s not found" % argv[0])
    if p.returncode != 0:
        raise ProjectError((p.stderr or p.stdout or "").strip()[-1500:] or
                           "helper exited %d" % p.returncode)
    return p.stdout
