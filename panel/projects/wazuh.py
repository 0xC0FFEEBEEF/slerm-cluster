"""Wazuh alert analysis -- one array task per normalised JSONL byte range."""
import os

import config
import db
import slurmapi
from projects.base import Project, ProjectError


DATA = config.WAZUH_DIR
SCRIPTS = config.WAZUH_SCRIPTS


def _report_id(value):
    try:
        report_id = int(value)
    except (TypeError, ValueError):
        raise ProjectError("a Wazuh report id is required")
    if report_id < 1:
        raise ProjectError("invalid Wazuh report id %r" % value)
    return report_id


def _report_id_from_list(list_path):
    name = os.path.basename(list_path)
    stem, ext = os.path.splitext(name)
    if ext != ".txt" or not stem.isdigit():
        raise ProjectError("unexpected Wazuh shard list %r" % name)
    return _report_id(stem)


def _list_path(project, report_id):
    path = os.path.realpath(os.path.join(project.lists_dir,
                                         "%d.txt" % report_id))
    root = os.path.realpath(project.lists_dir) + os.sep
    if not path.startswith(root):
        raise ProjectError("Wazuh list is outside the project list directory")
    return path


def _positive_int(value, label):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ProjectError("%s must be a positive integer" % label)
    if number < 1:
        raise ProjectError("%s must be a positive integer" % label)
    return number


def make_list(project, params):
    """Return the shard plan already written by wazuh.plan_shards()."""
    report_id = _report_id((params or {}).get("report"))
    if not db.get_wazuh_report(report_id):
        raise ProjectError("no such Wazuh report %d" % report_id)

    path = _list_path(project, report_id)
    try:
        with open(path) as fh:
            count = sum(1 for line in fh if line.strip())
    except OSError as exc:
        raise ProjectError("cannot read Wazuh shard list %s: %s" % (path, exc))
    if count < 1:
        raise ProjectError("Wazuh shard list %s is empty" % path)
    return path, count


def sbatch_args(project, list_path, count, options):
    """Build the primary Wazuh shard-array submission."""
    options = options or {}
    count = _positive_int(count, "shard count")
    throttle = _positive_int(
        options.get("throttle") or project.default_throttle, "throttle")
    throttle = min(throttle, 1000)
    partition = options.get("partition") or project.default_partition
    timelimit = options.get("timelimit") or project.default_timelimit
    mem = options.get("mem") or project.default_mem

    last = count - 1
    limit_value = options.get("limit")
    if limit_value not in (None, ""):
        limit = _positive_int(limit_value, "task limit")
        if limit < count:
            raise ProjectError(
                "Wazuh analysis cannot limit shards; all %d planned shards "
                "are required" % count)

    report_id = _report_id_from_list(list_path)
    shard_dir = os.path.join(DATA, "shards", str(report_id))
    os.makedirs(shard_dir, exist_ok=True)

    # Command-line resource flags override the defaults embedded in the shell
    # script, keeping the panel authoritative for each submission.
    return [
        "--partition=%s" % partition,
        "--time=%s" % timelimit,
        "--mem=%s" % mem,
        "--job-name=%s" % project.job_name,
        "--array=0-%d%%%d" % (last, throttle),
        project.array_script,
        list_path,
        shard_dir,
        config.DISPLAY_TZ,
    ]


def _export_path(list_path):
    try:
        with open(list_path) as fh:
            first = next((line.strip() for line in fh if line.strip()), "")
    except OSError as exc:
        raise ProjectError("cannot read Wazuh shard list %s: %s"
                           % (list_path, exc))
    parts = first.split()
    if len(parts) != 3:
        raise ProjectError("malformed Wazuh shard list %s" % list_path)
    return parts[0]


def post_submit(project, run_id, job_id, list_path, options):
    """Queue the global reduce after the shard array, even if a task fails."""
    run = db.get_run(run_id)
    if not run:
        raise ProjectError("no such run %d" % run_id)
    report_id = _report_id(run.get("params", {}).get("report"))
    report = db.get_wazuh_report(report_id)
    if not report:
        raise ProjectError("no such Wazuh report %d" % report_id)

    shard_dir = os.path.join(DATA, "shards", str(report_id))
    export_path = _export_path(list_path)
    report_path = (report.get("report_path") or
                   os.path.join(DATA, "reports", "%d.json" % report_id))
    partition = (options or {}).get("partition") or project.default_partition
    args = [
        "--dependency=afterany:%d" % job_id,
        "--partition=%s" % partition,
        "--mem=2G",
        "--job-name=wazuh-reduce",
        os.path.join(SCRIPTS, "wazuh_reduce.sh"),
        shard_dir,
        export_path,
        report_path,
        config.DISPLAY_TZ,
    ]
    try:
        reduce_id, _submit_line = slurmapi.submit(args, cwd=project.scripts_dir)
    except slurmapi.SlurmError as exc:
        db.update_wazuh_report(
            report_id, run_id=run_id, state="failed",
            message="array job %d submitted, but reduce submission failed: %s"
                    % (job_id, exc))
        raise
    db.update_wazuh_report(
        report_id, run_id=run_id, reduce_job_id=reduce_id, state="submitted",
        message="array job %d submitted; reduce job %d is waiting"
                % (job_id, reduce_id))


PROJECT = Project(
    id="wazuh",
    label="Wazuh",
    description="Wazuh alert snapshots analysed with global self-baselines. "
                "One array task builds mergeable counters for each byte range, "
                "then a dependent job reduces them into one report.",
    python="/usr/bin/python3",
    scripts_dir=SCRIPTS,
    data_dir=DATA,
    lists_dir=os.path.join(DATA, "lists"),
    out_dir=os.path.join(DATA, "reports"),
    logs_dir=os.path.join(DATA, "logs"),
    cache_dir=os.path.join(DATA, "reports"),
    array_script=os.path.join(SCRIPTS, "wazuh_array.sh"),
    job_name="wazuh",
    view="wazuh",
    products=[],
    default_partition="compute",
    default_timelimit="00:20:00",
    default_throttle=32,
    default_mem="1G",
    params=[],
    make_list=make_list,
    sbatch_args=sbatch_args,
    post_submit=post_submit,
)
