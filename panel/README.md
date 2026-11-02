# slerm-panel

Internal control panel for the slerm cluster. Runs on `slerm-head` as `slerm`,
serves on port 8080.

    http://slerm-head:8080

## Managing it

No root needed for any of this:

    systemctl --user restart slerm-panel
    systemctl --user status  slerm-panel
    journalctl --user-unit=slerm-panel -f

Note the log command: this host has no persistent per-user journal, so the
service's output lands in the system journal. `journalctl --user -u ...`
reports "No journal files were found"; `--user-unit=` is the form that works.

The code lives in `/home/slerm/slerm-panel`, which is also reachable from the
desktop at `/mnt/slerm/home/slerm/slerm-panel` over NFS -- edit there, then
restart the service.

## What is where

    app.py         Flask routes + waitress entrypoint
    slurmapi.py    wrappers around sinfo/scontrol/squeue/sacct --json, srun probes
    runs.py        listing -> sbatch -> progress tracking, log tails, date tokens
    scheduler.py   the recurrence thread (croniter); cluster scrontab is disabled
    gifbuild.py    Pillow GIF encoder + on-disk cache
    db.py          sqlite (runs, schedules, gpu probes) -> panel.db
    projects/      one module per project + the registry
    templates/, static/    the single-page UI

## Wazuh analysis

The Wazuh tab pulls alerts from one or more indexers (the home `nbdevlab` one and a
`remote` one reached through a tunnel), persists a
normalised JSONL snapshot, and analyses it as a Slurm array.  A CSV/JSON drop
zone provides the same analysis path when an indexer is unavailable.

The pipeline is deliberately two-pass:

    fetch/import -> export.jsonl -> array counter sidecars -> dependent reduce -> report

Array tasks only emit mergeable counters.  The reduce job computes one global
self-baseline over the entire requested window and then runs the detectors;
scoring inside a shard would make the answer change with the shard count.  The
report is written atomically, so its appearance under `/data/wazuh/reports` is
the success signal. Reducer errors publish an atomic `.failed` sibling, and the
panel tracks the dependent Slurm job so cancellation, timeout, OOM, and node
failure cannot leave a report stuck at `submitted`.

Indexer credentials are never stored in this directory or returned to the
browser.  Put read-only, `wazuh-alerts-*`-scoped profiles in
`~/.config/slerm-panel/wazuh.json` on `slerm-head`, mode `0600`:

    {
      "nbdevlab": {"url": "https://wuzah:9200", "user": "panel-ro", "password": "...", "verify": false},
      "remote": {"url": "https://cft:19201", "user": "panel-ro", "password": "...", "verify": false}
    }

The `remote` indexer is reached through `cft`, an always-on VM that proxies a local port to it; the
socket is allowed through UFW only from `slerm-head`. The UI distinguishes an unreachable remote path from a
genuinely quiet alert window.
Dashboard CSV exports containing exactly 10,000 rows are marked truncated
because that export path silently caps them.

Cluster-side code lives in `/data/wazuh/scripts`; snapshots, shard sidecars,
reports, logs, and uploads remain under `/data/wazuh`.  After changing shard or
detector code, run the shard-invariance self-test under the node's Python 3.9:

    /usr/bin/python3 /data/wazuh/scripts/wazuh_reduce.py \
      --selftest /data/wazuh/snapshots/import/4/export.jsonl \
      --tz America/Chicago

## Adding a project

Write `projects/<name>.py` exposing a `PROJECT = Project(...)` with two hooks:

    make_list(project, params)  -> (list_file_path, item_count)
    sbatch_args(project, list_path, count, options) -> [sbatch argv]

then add it to `REGISTRY` in `projects/__init__.py` and restart. The tab, the
run form (built from `params`), the frame viewer, the GIF builder and the
scheduler all pick it up with no UI changes. `projects/nexrad.py` is the
worked example.

## Notes

* No authentication. The firewall limits 8080 to 192.0.2.0/24; anything on that
  subnet can submit and cancel jobs as `slerm`.
* Node health comes from Slurm because the compute nodes are not ssh-reachable
  from the head. GPU temperature/utilisation therefore needs the "probe"
  button, which runs a one-minute `srun` job on that node -- it is on demand,
  never polled.
* Schedules support `{{date:yesterday}}`, `{{date:today}}` and
  `{{date:today-N}}` in parameters, expanded when the schedule fires.

## Products

The NOAA tab holds several derived series, each with its own output directory,
frame index and GIF cache, all sharing one dataset selector and one time
cursor (anchored to a frame *name*, so panes stay on the same scan even when a
product is only part-rendered). Add one by appending a `Product` in
`projects/nexrad.py`.

    reflectivity  base PPI                      process_scan.py
    rotation      azimuthal shear over refl.    process_rotation.py    3G
    storms        cells + centroids + sidecars  process_storms.py      3G
    vad           per-scan wind profiles        process_vad.py         3G
                  assembled by                  assemble_vad.py
    wind          CONUS 10 m wind from HRRR     process_hrrr_wind.py   4G

Memory is per product because the footprints differ a lot -- region-based
dealiasing peaks at ~2 GB, against ~1 GB for a plain render. The run form's
memory field follows the product selection so a rotation run cannot be
submitted at the reflectivity default and get OOM-killed.

### Three things that quietly produce wrong answers

* **Split cuts.** NEXRAD scans the lowest elevation twice: a surveillance cut
  (reflectivity + dual-pol) and a Doppler cut (velocity + spectrum width).
  Sweep 0 has *no velocity at all*, and the Doppler cut has *no RhoHV*. Use
  `lowest_velocity_sweep()`, never sweep 0.
* **Azimuthal shear is not `np.gradient`.** Arc length shrinks toward the
  radar, so a plain derivative peaked at 0.91/s two kilometres out and called
  52k gates mesocyclonic. `azimuthal_shear()` is the Smith & Elmore LLSD fit
  with smoothing and a range window.
* **A reflectivity floor does not reject biology.** Nocturnal insect and bird
  return runs 10-25 dBZ and fills the area near the radar; with only a 15 dBZ
  floor the "strongest rotation of the day" was bio-scatter, reading 0.0418/s
  where the real value was 0.0047/s. `map_field_across_sweeps()` borrows RhoHV
  from the surveillance cut by nearest azimuth (worst gap measured: 0.09 deg)
  so the dual-pol mask works on the Doppler sweep.

* **Shear clusters form in clear air.** Even with the dual-pol mask, the dense
  sampling and 1/r term near the radar manufacture above-threshold specks on
  scans with no convection anywhere. Ranking by the hottest single gate made a
  4-gate artefact the "peak rotation of the day". `process_rotation.py` now
  keeps only contiguous clusters (>= 6 gates) that sit on or beside convective
  reflectivity (>= 35 dBZ, dilated), and reports the peak within those. On a
  test scan with 3 convective gates it rejected all 4 clusters; on one with
  1953 it kept a genuine 0.0271/s couplet.

VAD has its own trap: the fit goes ill-conditioned where coverage is sparse and
returns speeds of several hundred m/s. Both the retrieval and the assembler
reject anything above 75 m/s, and the colour scale is capped rather than
autoranged.

## Why the array does not ask for GPUs

`process_scan.py` is an S3 fetch, a Py-ART decode and a matplotlib render --
none of that touches CUDA, and Py-ART has no GPU backend. Requesting
`--gres=gpu:1` would make every task hold a card while doing pure CPU work,
blocking real GPU jobs and speeding up nothing.

What the gpu partition is actually worth here is its 12 CPU cores, so the
array runs on `compute,gpu` with no `--gres`. GPU jobs still get a card while
an array is in flight; they may wait a little for cores.

The bigger win was memory. Measured peak RSS over a full 218-scan day is
998 MB against the old 2G request, which capped the compute nodes at 7 tasks
each. At `--mem=1400M`:

    node01/02   16 cores, 15729 MB -> 11 tasks each
    node03/04    6 cores,  9614 MB ->  6 tasks each (cpu-bound)
    total 34 concurrent, up from 14

Both are panel-side options (`default_partition`, `default_mem` in
`projects/nexrad.py`, overridable per run in the form), so nothing in
`nexrad_array.sh` needed editing -- the sbatch command line wins over the
`#SBATCH` directives in the script.

## Time zones

The NOAA archive is UTC end to end: S3 prefixes are UTC days and every scan
filename is a UTC timestamp. Two things are shown in local time instead:

* the panel labels scan times in `config.DISPLAY_TZ` (`America/Chicago`,
  DST-aware), parsed from the filename by `projects/nexrad.py:time_of`;
* `/apps/nexrad/scripts/process_scan.py` writes the same zone into the title
  it burns into each PNG. The pre-conversion script is kept alongside it as
  `process_scan.py.utc-backup`.

Those two must agree -- if you change one, change the other, and re-render the
affected frames so old PNGs do not keep a stale label.

### Picking a day to run

The run form takes a day plus the zone that day is measured in, defaulting to
Central. They are genuinely different requests:

    2024-05-26 UTC      -> one prefix,  218 scans
    2024-05-26 Central  -> two prefixes, 201 scans
                           (2024-05-26T05:00Z .. 2024-05-27T05:00Z)

In Central mode `make_list` lists every UTC prefix the local day touches and
then trims to the exact window, so the first and last scans really are
00:05 and 23:55 local. The boundaries come from local midnight to local
midnight rather than "start + 24h", which keeps the 23- and 25-hour DST days
right (verified against 2024-03-10 and 2024-11-03). List files are suffixed
with the mode -- `kdgx-20240526-central.txt` vs `-utc.txt` -- so the two never
overwrite each other, and `/api/p/<id>/resolve-day` returns the same window
the form previews.

What stays UTC regardless: the frame filenames (that is how the archive names
them) and the dataset grouping in the animation. A Central-day run therefore
produces frames in two dataset groups; if that becomes annoying, the fix is a
Central-day `group_of` in `projects/nexrad.py`.
* GIFs are cached under each project's `cache_dir` (`/data/nexrad/gif`) and
  keyed on the frame list plus encoding settings, so new renders invalidate
  the cache automatically. `gifbuild.prune()` trims old ones.
* The output directory accumulates across runs, so the Dataset selector groups
  frames into one radar site + UTC day (`projects/nexrad.py:group_of`) and
  drives both the animation and the frame scrubber. "all frames" is still
  there. A project without a `group_of` hook simply has no selector.
