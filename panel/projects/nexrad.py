"""NOAA NEXRAD Level II -- the project currently running on the cluster.

The panel orchestrates the existing scripts in /apps/nexrad/scripts; it does
not replace or modify them.

  list_scans.py <YYYY/MM/DD> <SITE> <out.txt>   enumerate S3 keys for a day
  nexrad_array.sh <list> <outdir>               one array task per scan,
                                                each calling process_scan.py
"""
import datetime
import os
import re
import tempfile
from zoneinfo import ZoneInfo

import config

from projects.base import Param, Product, Project, ProjectError, run_helper

VENV_PY = "/apps/nexrad/venv/bin/python"
SCRIPTS = "/apps/nexrad/scripts"
DATA = "/data/nexrad"

DATE_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")
DASHED_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
SITE_RE = re.compile(r"^[A-Z0-9]{4}$")
# KDGX20240526_000156_V06.png -> site KDGX, day 2024-05-26, 00:01:56 UTC
FRAME_RE = re.compile(r"^([A-Z0-9]{4})(\d{4})(\d{2})(\d{2})_")
FRAME_TIME_RE = re.compile(r"^[A-Z0-9]{4}(\d{8})_(\d{6})")


def _parse_day(value):
    """Accept YYYY-MM-DD (date picker) or YYYY/MM/DD (token / typed)."""
    value = (value or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ProjectError("day must look like YYYY-MM-DD (got %r)" % value)


def _key_time(key):
    """UTC timestamp encoded in an archive key, e.g. .../KDGX20240526_000156_V06."""
    match = FRAME_TIME_RE.match(key.rsplit("/", 1)[-1])
    if not match:
        return None
    try:
        stamp = datetime.datetime.strptime(match.group(1) + match.group(2),
                                           "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return stamp.replace(tzinfo=datetime.timezone.utc)


def resolve_day(day, mode):
    """Work out which UTC prefixes to list, and the window to keep.

    A Central day is not a UTC day: 2024-05-26 in Central runs from 05:00Z on
    the 26th to 05:00Z on the 27th, so it spans two of the bucket's prefixes
    and needs trimming at both ends. Using local midnight-to-midnight (rather
    than adding 24h) keeps the 23- and 25-hour DST days correct.
    """
    if mode == "utc":
        return [day], None

    tz = ZoneInfo(config.DISPLAY_TZ)
    start_local = datetime.datetime.combine(day, datetime.time(0, 0), tzinfo=tz)
    end_local = datetime.datetime.combine(day + datetime.timedelta(days=1),
                                          datetime.time(0, 0), tzinfo=tz)
    window = (start_local.astimezone(datetime.timezone.utc),
              end_local.astimezone(datetime.timezone.utc))

    prefixes = []
    cursor = window[0].date()
    last = (window[1] - datetime.timedelta(seconds=1)).date()
    while cursor <= last:
        prefixes.append(cursor)
        cursor += datetime.timedelta(days=1)
    return prefixes, window


def day_info(date_str, mode="central"):
    """Describe the UTC window a requested day resolves to, for the UI hint."""
    day = _parse_day(date_str)
    mode = (mode or "central").strip().lower()
    if mode not in ("central", "utc"):
        raise ProjectError("date_tz must be 'central' or 'utc'")
    prefixes, window = resolve_day(day, mode)
    info = {
        "day": day.isoformat(),
        "mode": mode,
        "tz": config.DISPLAY_TZ,
        "prefixes": [p.strftime("%Y/%m/%d") for p in prefixes],
    }
    if window:
        info["window_start"] = window[0].isoformat().replace("+00:00", "Z")
        info["window_end"] = window[1].isoformat().replace("+00:00", "Z")
        info["hours"] = round((window[1] - window[0]).total_seconds() / 3600, 1)
    return info


def make_list(project, params):
    """Enumerate a day of scans for one radar site into a list file."""
    site = (params.get("site") or "").strip().upper()
    if not SITE_RE.match(site):
        raise ProjectError("site must be a 4-character radar id like KDGX "
                           "(got %r)" % site)
    day = _parse_day(params.get("date"))
    mode = (params.get("date_tz") or "central").strip().lower()
    if mode not in ("central", "utc"):
        raise ProjectError("date_tz must be 'central' or 'utc' (got %r)" % mode)

    project.ensure_dirs()
    prefixes, window = resolve_day(day, mode)

    keys = []
    for prefix in prefixes:
        # unique per call: two runs listing the same site/day at once (say
        # reflectivity and rotation) would otherwise share this scratch file
        # and delete it out from under each other.
        fd, part = tempfile.mkstemp(
            dir=project.lists_dir, suffix=".txt",
            prefix=".part-%s-%s-" % (site.lower(), prefix.strftime("%Y%m%d")))
        os.close(fd)
        try:
            run_helper([VENV_PY, os.path.join(SCRIPTS, "list_scans.py"),
                        prefix.strftime("%Y/%m/%d"), site, part], timeout=600)
            with open(part) as fh:
                keys.extend(line.strip() for line in fh if line.strip())
        finally:
            if os.path.exists(part):
                os.unlink(part)

    if window:
        kept = []
        for key in keys:
            stamp = _key_time(key)
            if stamp is not None and window[0] <= stamp < window[1]:
                kept.append(key)
        keys = kept
    keys = sorted(set(keys))

    list_path = os.path.join(
        project.lists_dir,
        "%s-%s-%s.txt" % (site.lower(), day.strftime("%Y%m%d"), mode))
    # write-then-rename so a concurrent reader never sees a half-written list
    fd, staged = tempfile.mkstemp(dir=project.lists_dir, suffix=".txt",
                                  prefix=".staging-")
    with os.fdopen(fd, "w") as fh:
        for key in keys:
            fh.write(key + "\n")
    os.replace(staged, list_path)

    if not keys:
        raise ProjectError(
            "no scans found for %s on %s (%s) -- nothing to submit. "
            "Checked UTC prefix(es): %s"
            % (site, day.isoformat(), mode.upper(),
               ", ".join(p.strftime("%Y/%m/%d") for p in prefixes)))
    return list_path, len(keys)


def group_of(name):
    """One dataset per radar site per UTC day, read off the frame filename."""
    match = FRAME_RE.match(name)
    if not match:
        return None
    site, year, month, day = match.groups()
    return ("%s-%s%s%s" % (site, year, month, day),
            "%s \u00b7 %s-%s-%s" % (site, year, month, day))


def time_of(name):
    """Scan time from the filename, as UTC epoch seconds.

    The archive names every scan in UTC; the panel renders these in
    config.DISPLAY_TZ so they line up with the plot titles.
    """
    match = FRAME_TIME_RE.match(name)
    if not match:
        return None
    try:
        stamp = datetime.datetime.strptime(match.group(1) + match.group(2),
                                           "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return int(stamp.replace(tzinfo=datetime.timezone.utc).timestamp())


def sbatch_args(project, list_path, count, options):
    """Build the sbatch command line for the array over that list."""
    throttle = int(options.get("throttle") or project.default_throttle)
    throttle = max(1, min(throttle, 1000))
    partition = options.get("partition") or project.default_partition
    timelimit = options.get("timelimit") or project.default_timelimit
    product_for_mem = project.product(options.get("product"))
    mem = options.get("mem") or product_for_mem.default_mem or project.default_mem
    limit = options.get("limit")
    last = count - 1
    if limit:
        last = min(last, int(limit) - 1)

    product = project.product(options.get("product"))
    if not product.script:
        raise ProjectError("product %r is not rendered per scan" % product.id)

    # --mem on the command line overrides the #SBATCH line in the array script,
    # so the panel stays authoritative without editing the script.
    return [
        "--partition=%s" % partition,
        "--time=%s" % timelimit,
        "--mem=%s" % mem,
        "--job-name=%s-%s" % (project.job_name, product.id),
        "--array=0-%d%%%d" % (last, throttle),
        project.array_script,
        list_path,
        product.out_dir,
        product.script,
    ]


def _product(pid, label, description, script=None, kind="series",
             overlays=None, mem=None, assemble=None):
    return Product(
        id=pid, label=label, description=description,
        out_dir=os.path.join(DATA, "products", pid),
        cache_dir=os.path.join(DATA, "gif", pid),
        script=os.path.join(SCRIPTS, script) if script else None,
        kind=kind, group_of=group_of, time_of=time_of, overlays=overlays,
        default_mem=mem, assemble=assemble,
    )


HRRR_RE = re.compile(r"^hrrr_(\d{8})_(\d{2})z")


def wind_group_of(name):
    """HRRR frames group by UTC day, same idea as the radar products."""
    match = HRRR_RE.match(name)
    if not match:
        return None
    day = match.group(1)
    return ("HRRR-%s" % day,
            "CONUS wind \u00b7 %s-%s-%s" % (day[:4], day[4:6], day[6:]))


def wind_time_of(name):
    match = HRRR_RE.match(name)
    if not match:
        return None
    stamp = datetime.datetime.strptime(match.group(1) + match.group(2),
                                       "%Y%m%d%H")
    return int(stamp.replace(tzinfo=datetime.timezone.utc).timestamp())


def wind_make_list(project, product, params):
    """One HRRR analysis hour per work item, covering the requested day."""
    day = _parse_day(params.get("date"))
    mode = (params.get("date_tz") or "central").strip().lower()
    prefixes, window = resolve_day(day, mode)

    if window:
        start, end = window
    else:
        start = datetime.datetime.combine(day, datetime.time(0, 0),
                                          tzinfo=datetime.timezone.utc)
        end = start + datetime.timedelta(days=1)

    stamps = []
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor < end:
        stamps.append(cursor.strftime("%Y%m%d_%Hz"))
        cursor += datetime.timedelta(hours=1)

    project.ensure_dirs()
    list_path = os.path.join(project.lists_dir,
                             "hrrr-%s-%s.txt" % (day.strftime("%Y%m%d"), mode))
    fd, staged = tempfile.mkstemp(dir=project.lists_dir, suffix=".txt",
                                  prefix=".staging-")
    with os.fdopen(fd, "w") as fh:
        for stamp in stamps:
            fh.write(stamp + "\n")
    os.replace(staged, list_path)
    if not stamps:
        raise ProjectError("no analysis hours in that window")
    return list_path, len(stamps)


def vad_assemble(project, product, group):
    """Turn the per-scan VAD sidecars into one time-height panel."""
    if not group or group == "all":
        raise ProjectError("pick a dataset first -- a wind profile covers one "
                           "site and day, not everything at once")
    try:
        site, day = group.split("-", 1)
    except ValueError:
        raise ProjectError("unexpected dataset key %r" % group)

    product.ensure_dirs()
    # named so it lands in the same dataset group as the scans it came from
    out_path = os.path.join(product.out_dir, "%s%s_000000_VAD.png" % (site, day))
    run_helper([VENV_PY, os.path.join(SCRIPTS, "assemble_vad.py"),
                product.out_dir, out_path, "%s%s" % (site, day)], timeout=600)
    return out_path


PRODUCTS = [
    Product(
        id="reflectivity", label="Reflectivity",
        description="Base reflectivity PPI at the lowest sweep -- what the "
                    "storm looks like.",
        out_dir=os.path.join(DATA, "out"),          # the original location
        cache_dir=os.path.join(DATA, "gif"),
        script=os.path.join(SCRIPTS, "process_scan.py"),
        group_of=group_of, time_of=time_of,
    ),
    _product("rotation", "Rotation",
             "Azimuthal shear from dealiased velocity, contoured over "
             "reflectivity. Mesocyclone signatures show as tight couplets.",
             script="process_rotation.py", overlays="reflectivity", mem="3G"),
    _product("storms", "Storm cells",
             "Identified cells with centroids and the track each one has "
             "followed, over reflectivity.",
             script="process_storms.py", overlays="reflectivity", mem="3G"),
    _product("vad", "Wind profile",
             "VAD-retrieved wind speed and direction versus height over the "
             "radar. Each scan contributes one profile; assemble them into a "
             "time-height panel for the day.",
             script="process_vad.py", kind="series", mem="3G",
             assemble=vad_assemble),
    Product(
        id="wind", label="US wind",
        description="CONUS 10 m wind from the HRRR analysis -- speed shading "
                    "with streamlines, one frame per analysis hour.",
        note="A single radar only measures motion along its own beam, so a "
             "wind field cannot come from NEXRAD. This is model analysis data "
             "from the HRRR-Zarr archive.",
        out_dir=os.path.join(DATA, "products", "wind"),
        cache_dir=os.path.join(DATA, "gif", "wind"),
        script=os.path.join(SCRIPTS, "process_hrrr_wind.py"),
        group_of=wind_group_of, time_of=wind_time_of,
        make_list=wind_make_list, default_mem="4G",
    ),
]


PROJECT = Project(
    id="nexrad",
    label="NOAA",
    description="NEXRAD Level II reflectivity renders from the NOAA open-data "
                "bucket on S3. One array task per radar scan.",
    python=VENV_PY,
    scripts_dir=SCRIPTS,
    data_dir=DATA,
    lists_dir=os.path.join(DATA, "lists"),
    out_dir=os.path.join(DATA, "out"),
    logs_dir=os.path.join(DATA, "logs"),
    cache_dir=os.path.join(DATA, "gif"),
    array_script=os.path.join(SCRIPTS, "nexrad_product.sh"),
    products=PRODUCTS,
    job_name="nexrad",
    # Both partitions: the pipeline is S3 fetch + Py-ART decode + matplotlib,
    # none of which touches CUDA, so the gpu nodes are worth having for their
    # 12 otherwise-idle cores. Deliberately no --gres, so real GPU jobs can
    # still get a card while an array is running.
    default_partition="compute,gpu",
    default_timelimit="00:15:00",
    default_throttle=64,
    # Measured peak RSS over a full 218-scan day was 998 MB; 2G was capping
    # concurrency at 7 tasks/node on the compute nodes for no reason.
    default_mem="1400M",
    params=[
        Param(name="site", label="Radar site", default="KDGX",
              placeholder="KDGX",
              help="Four-character NEXRAD site id, e.g. KDGX (Jackson MS)."),
        Param(name="date", label="Day", default="2024/05/26",
              placeholder="YYYY-MM-DD", kind="day",
              help="A Central day spans two UTC prefixes in the bucket; the "
                   "list is trimmed to the exact local window."),
    ],
    make_list=make_list,
    sbatch_args=sbatch_args,
    day_info=day_info,
)
