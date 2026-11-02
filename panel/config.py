"""Slerm Panel configuration."""
import os

BIND_HOST = os.environ.get("PANEL_HOST", "0.0.0.0")
BIND_PORT = int(os.environ.get("PANEL_PORT", "8080"))
THREADS = int(os.environ.get("PANEL_THREADS", "8"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("PANEL_DB", os.path.join(BASE_DIR, "panel.db"))

# how far back the cluster history view reaches by default
HISTORY_DAYS = 3

# scheduler tick
TICK_SECONDS = 20

# How often the browser re-polls each thing, in milliseconds. The UI ticks at
# POLL["tick"] and each target is due on its own interval, so the cheap live
# views update every second without dragging the expensive ones with them.
POLL = {
    "tick": 1000,        # base UI tick
    "cluster": 1000,     # nodes, queue, head stats
    "history": 20000,    # sacct -- the heaviest query here
    "project": 1000,     # project status tiles + run list
    "runProgress": 2000, # per-run squeue+sacct task breakdown
    "frames": 5000,      # frame index (a few hundred entries)
    "gif": 8000,         # cache state when idle; a build polls faster itself
    "schedules": 3000,
    "wazuh": 2000,
    "ai": 5000,          # rollups over the request ledger plus a server probe
    "aiLive": 1500,      # progress bars only: /v1/models + /slots, cached ~1s
}

# slurm command timeouts (seconds)
SLURM_TIMEOUT = 20
SRUN_PROBE_TIMEOUT = 120

# Scan timestamps in the NOAA archive are UTC; the panel labels them in this
# zone so the UI agrees with the title burned into each rendered PNG by
# /apps/nexrad/scripts/process_scan.py. Change both together.
DISPLAY_TZ = os.environ.get("PANEL_TZ", "America/Chicago")

# filesystems shown on the cluster tab
WATCHED_MOUNTS = ["/data", "/home", "/apps"]

# --- Wazuh -------------------------------------------------------------------
WAZUH_DIR = "/data/wazuh"
WAZUH_SCRIPTS = os.path.join(WAZUH_DIR, "scripts")
# Credentials live outside the panel directory, mode 0600, and are never
# returned to the browser.
WAZUH_PROFILES_PATH = os.environ.get(
    "PANEL_WAZUH_PROFILES",
    os.path.expanduser("~/.config/slerm-panel/wazuh.json"))
WAZUH_INDEX_PATTERN = "wazuh-alerts-4.x-*"
WAZUH_TIMEOUT = 60
WAZUH_HEALTH_TIMEOUT = 6
# 5000 reduced into a buffer large enough to trip the indexer's parent
# circuit breaker on the home node (1 GB heap). 1000 pages comfortably.
# 2000 measured fastest over a WAN path: 1156 alerts/s vs 731 at 1000
# and 970 at 4000, where the per-page transfer starts to dominate.
WAZUH_PAGE_SIZE = 2000
# Bounded total-hit tracking; an exact count is a full cross-shard reduce.
WAZUH_TRACK_TOTAL_HITS = 50000
# Fields no detector reads, dropped at the indexer so they are never
# transferred, parsed or stored. The compliance mappings alone are six fields
# of pure regulatory cross-reference; data.win.system.message is the rendered
# human-readable Windows event, redundant with data.win.eventdata.*, and is the
# single largest field on a Windows estate. full_log is deliberately KEPT --
# it is what finding evidence is made of.
WAZUH_SOURCE_EXCLUDES = [
    "rule.pci_dss", "rule.gdpr", "rule.hipaa", "rule.nist_800_53",
    "rule.gpg13", "rule.tsc", "rule.cis", "rule.cis_csc_v7",
    "rule.cis_csc_v8", "rule.soc_2", "rule.mitre.technique",
    "data.win.system.message", "data.win.eventdata.binary",
    "previous_output",
    # Measured on a 2,000-alert sample (1,226 B/alert): registry FIM
    # dominates the estate and these carry none of the signal. win_perm_*
    # alone is 14.5% of every byte fetched -- ACL blobs no human reads -- and
    # the before/after digests another ~8%. What a FIM finding needs is THAT
    # the object changed and its path, both of which survive. full_log (25.4%)
    # is deliberately kept: it is the raw line the evidence pane shows.
    "syscheck.win_perm_after", "syscheck.win_perm_before",
    "syscheck.sha1_after", "syscheck.sha1_before",
    "syscheck.md5_after", "syscheck.md5_before",
    "syscheck.sha256_before",
]
# Paging grows on success and halves on an indexer overload, so a healthy
# cluster is not permanently punished for one circuit-breaker trip.
WAZUH_PAGE_MAX = 4000
# A month of the larger indexer is 3.85M alerts, so 2M silently halved the window it was
# asked for. `capped` is returned and surfaced, but the right cap is one a
# normal request does not hit.
WAZUH_MAX_ALERTS = 6_000_000
WAZUH_UPLOAD_MAX_BYTES = 512 * 1024 * 1024
# Aim for this much normalised JSONL per array task.
WAZUH_SHARD_BYTES = 16 * 1024 * 1024
WAZUH_MAX_SHARDS = 256

# Disjoint time slices pulled concurrently. Paging within a slice stays
# sequential (search_after is a cursor); the parallelism is across windows.
WAZUH_FETCH_SLICES = int(os.environ.get("PANEL_WAZUH_SLICES", "4"))

# Concurrent in-flight indexer requests across all fetch slices. The parent
# circuit breaker trips on total committed heap, so this - not page size - is
# the lever that matters on a small-heap indexer. Shrinks automatically on
# overload and never grows back within a fetch.
# Throughput plateaus at ~2000 alerts/s: 2 threads reached 2048/s and 8 only
# 2143/s, so the ceiling is the WAN link, not indexer concurrency. Going
# wider only risks another circuit-breaker trip for no gain.
WAZUH_MAX_INFLIGHT = int(os.environ.get("PANEL_WAZUH_INFLIGHT", "3"))

# --- optional LLM triage ------------------------------------------------------
# Two ways to reach a model. WAZUH_LLM_CLUSTER routes triage at the slerm
# cluster's own llama.cpp router, and that one CANNOT be a fixed URL: the router
# job moves to whichever GPU node Slurm hands it on every requeue, so a pinned
# host goes stale the first time the job is requeued. When it is on, the
# endpoint is resolved from the queue at call time through llmusage.servers() --
# the same discovery the AI tab uses -- and WAZUH_LLM_URL/TOKEN are ignored.
# Set PANEL_WAZUH_LLM_CLUSTER=0 to go back to a pinned OpenAI-compatible URL
# (LM Studio, Ollama, anything else).
WAZUH_LLM_CLUSTER = os.environ.get(
    "PANEL_WAZUH_LLM_CLUSTER", "1") not in ("0", "no", "false")

# Advisory only: annotations live in a sidecar and never alter, reorder or
# filter the deterministic findings. With WAZUH_LLM_CLUSTER off, an empty URL
# disables the feature entirely.
WAZUH_LLM_URL = os.environ.get("PANEL_WAZUH_LLM_URL", "")   # e.g. http://lmstudio:1234/v1
WAZUH_LLM_TOKEN = os.environ.get("PANEL_WAZUH_LLM_TOKEN", "")
WAZUH_LLM_MODEL = os.environ.get("PANEL_WAZUH_LLM_MODEL", "")
# ~90s per finding measured on cyclops; headroom for a cold model load.
WAZUH_LLM_TIMEOUT = int(os.environ.get("PANEL_WAZUH_LLM_TIMEOUT", "300"))
# Bonsai is a REASONING model: it spent 1,642 tokens thinking before writing
# 163 tokens of answer. At 600 the entire budget went to reasoning and the
# content came back EMPTY, surfacing only as "model did not return JSON".
WAZUH_LLM_MAX_TOKENS = int(os.environ.get("PANEL_WAZUH_LLM_MAX_TOKENS", "2500"))
# Only the top findings are triaged: a 27B model at ~20s per finding would take
# an hour over a 178-finding report, and the tail is where the noise lives.
WAZUH_LLM_TOP_N = int(os.environ.get("PANEL_WAZUH_LLM_TOP_N", "20"))

# Constrain triage output to RESPONSE_SCHEMA via LM Studio's json_schema
# response_format. Off only if you swap in a server that cannot do grammars.
# Model choice and generation settings are editable from the UI and saved
# here, layered over these defaults. Kept beside the indexer credentials and
# written 0600. The URL and token stay in llm.env: infrastructure and a
# secret, neither settable from a browser.
WAZUH_LLM_SETTINGS_PATH = os.environ.get(
    "PANEL_WAZUH_LLM_SETTINGS",
    os.path.join(os.path.expanduser("~/.config/slerm-panel"), "llm.json"))
# Whether the model is allowed its reasoning turn. Off sends
# chat_template_kwargs.enable_thinking=false; templates that ignore it will
# reason regardless, so treat this as a request rather than a guarantee.
# Public threat-intel lookup per finding. OFF by default and deliberately so:
# it sends a query to a third party, and these alerts describe a client's
# network. Only rule ids, rule description text and MITRE tactic names are
# ever sent -- see wazuhweb.terms_for.
# Triage every ranked finding rather than the top N. On by default: the cap
# existed because a finding took ~90s, and at ~11s it no longer earns its
# keep. Routine/demoted findings are still never triaged -- "the whole list"
# means everything that survived suppression, not all several thousand rows.
WAZUH_LLM_ALL_FINDINGS = os.environ.get(
    "PANEL_WAZUH_LLM_ALL", "1") not in ("0", "no", "false")

WAZUH_LLM_WEB_SEARCH = os.environ.get(
    "PANEL_WAZUH_LLM_WEB_SEARCH", "0") not in ("0", "no", "false")

WAZUH_LLM_THINKING = os.environ.get(
    "PANEL_WAZUH_LLM_THINKING", "1") not in ("0", "no", "false")
WAZUH_LLM_TEMPERATURE = float(os.environ.get("PANEL_WAZUH_LLM_TEMPERATURE", "0.1"))

WAZUH_LLM_JSON_SCHEMA = os.environ.get(
    "PANEL_WAZUH_LLM_JSON_SCHEMA", "1") not in ("0", "no", "false")

# Start triage automatically the moment a report is ready, so the AI
# fields populate as they are produced instead of on demand.
WAZUH_LLM_AUTO = os.environ.get("PANEL_WAZUH_LLM_AUTO", "1") not in ("0", "no", "false")


# --- cluster inference accounting (the AI tab) --------------------------------
# What the GPU nodes were asked to generate, for every client of them -- Open
# WebUI, the delegate harness, a curl, this panel. Read-only: the panel samples
# the servers' own logs and /metrics and never sends them work.
#
# Which Slurm jobs are inference servers. The router multiplexes several models
# behind one port; server.sh serves exactly one. Both log in the same format.
# llm-router-cpu: the orchestrator's CPU-only fallback router on deadlock.
LLM_SERVER_JOB_NAMES = ("llm-router", "llm-router-cpu", "llm-server")
# slerm orchestrator (~/slerm-orch): per-model availability for the catalogue,
# and its localhost control port for the "Bring up" button.
LLM_ORCH_STATE = "/data/llm/run/orchestrator.json"
LLM_ORCH_CONTROL = "http://127.0.0.1:8091"
# The router does not only use the card it is sitting on. router-rpc.sh submits
# or adopts separate Slurm jobs that expose OTHER nodes' GPUs and CPUs over
# ggml-rpc, and pools them into one device list -- that is how a 20B model fits
# on 6 GB cards. Those jobs are what the AI tab has to look at to show the whole
# pool rather than just the local card.
LLM_WORKER_JOB_NAMES = ("llm-rpcworker", "llm-rpccpu")
LLM_WORKER_CPU_NAMES = ("llm-rpccpu",)
LLM_SERVER_PORT = int(os.environ.get("PANEL_LLM_PORT", "8080"))
LLM_SERVER_TIMEOUT = int(os.environ.get("PANEL_LLM_TIMEOUT", "6"))
LLM_LOG_DIR = os.environ.get("PANEL_LLM_LOG_DIR", "/data/llm/logs")

# How often the sampler tails those logs. This bounds how precisely a request
# is timestamped -- llama.cpp's log clock cannot be turned into wall time (see
# llmlog.py), so a request is stamped when it is read. It also has to be short
# enough that LLM_LOG_MAX_READ is never the limit: the servers emit a progress
# line every ~3 seconds per active request.
LLM_POLL_SECONDS = int(os.environ.get("PANEL_LLM_POLL", "20"))
# Ceiling on one read, so a log that grew while the panel was down cannot pull
# an unbounded amount into memory. The remainder is read on the next poll.
LLM_LOG_MAX_READ = 4 * 1024 * 1024
LLM_KEEP_DAYS = int(os.environ.get("PANEL_LLM_KEEP_DAYS", "180"))
