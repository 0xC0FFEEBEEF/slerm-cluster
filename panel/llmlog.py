"""Parse llama.cpp server logs into completed-request records.

The cluster's inference servers (`/data/llm/jobs/*.sh`) write one log per Slurm
job under `/data/llm/logs`. That log is the only place a *per-request* record
exists: `/metrics` gives cumulative counters and the OpenAI API gives nothing
after the fact, so this is what "a model was called" is made of.

The router multiplexes several model instances into one file. Its own lines are
unprefixed; a child instance prefixes every line with its listening port:

    [48037] 59.37.558.037 I slot launch_slot_: id  0 | task 5068 | processing task
    [48037] 69.33.258.944 I slot print_timing: id  0 | task 5068 | prompt eval time = 29838.34 ms /   286 tokens (...)
    [48037] 69.33.258.956 I slot print_timing: id  0 | task 5068 |        eval time = 565862.50 ms /  2470 tokens (...)
    [48037] 69.33.258.960 I slot print_timing: id  0 | task 5068 |       total time = 595700.84 ms /  2756 tokens
    [48037] 69.33.260.471 I slot      release: id  0 | task 5068 | stop processing: n_tokens = 17576, truncated = 0

**The leading clock is deliberately ignored.** It is MM.SS.mmm.uuu since *that
stream* started, and each child has its own zero, so the file carries no shared
absolute time -- measured: the router's clock ran 601 s behind its own Slurm job
start, and a child's ran 8771 s behind the router's. Reconstructing wall time
from it means guessing. The caller stamps a record with the moment it was READ
instead, which is exact to the poll interval and needs no arithmetic; the price
is that a log is only ever tailed forward, never backfilled.

Nothing here reads the network or the database, so it can be tested against a
real log directly.
"""
import re

# "[48037] 59.37.558.037 I " -> port 48037. No prefix means the router itself.
_LINE = re.compile(r"^(?:\[(\d+)\]\s+)?[\d.]+\s+[A-Z]\s+(.*)$")

_TASK = re.compile(r"\btask\s+(\d+)\b")
_PROMPT_EVAL = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
_EVAL = re.compile(r"^\s*eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
_TOTAL = re.compile(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
_RELEASE = re.compile(
    r"stop processing:\s*n_tokens\s*=\s*(\d+),\s*truncated\s*=\s*(\d+)")
_PROXY = re.compile(
    r"proxying request to model\s+(\S+)\s+on port\s+(\d+)")
_LOADING = re.compile(r"model name=(\S+) is not loaded, loading")
_UNLOAD = re.compile(r"stopping model instance name=(\S+)")
_IDLE = re.compile(r"model name=(\S+) went idle")
_LOAD_MODEL = re.compile(r"load_model: loading model '([^']+)'")

# Partial requests kept while waiting for their closing lines. A request that
# never finishes would otherwise sit here for the life of the process.
MAX_PENDING = 500

# How many chunks a finished request waits for its `release` line before being
# emitted without one. Only the last request in a burst ever reaches this: a
# new task on the same slot closes the previous one immediately.
HOLD_CHUNKS = 3


class Parser(object):
    """Incremental parser. One instance per log file, kept across polls.

    A request straddles several lines and a poll can land in the middle of one,
    so partial records are held in `pending` until their `total time` line
    arrives rather than being dropped or emitted half-filled.
    """

    def __init__(self, port_models=None):
        self.pending = {}          # (port, task) -> partial record
        self.done = {}             # saw `total time`, waiting for `release`
        self.port_model = dict(port_models or {})   # port -> model alias
        self.events = []
        self.requests = []
        self._tail = ""            # bytes after the last newline of the chunk
        self._gen = 0              # chunks fed, for the flush backstop

    def model_for(self, port):
        return self.port_model.get(port)

    def feed(self, text):
        """Consume a chunk of log text. Returns (requests, events) found in it.

        A poll reads whatever bytes have landed, which routinely cuts the last
        line in half. That partial line is carried into the next chunk rather
        than parsed: splitting a `total time` line across two reads otherwise
        loses a whole request, silently -- measured at 31 of 32 requests when
        the same file was fed in 997-byte chunks.
        """
        self.requests = []
        self.events = []
        text = self._tail + text
        if text.endswith("\n"):
            self._tail = ""
        else:
            cut = text.rfind("\n")
            self._tail = text[cut + 1:]
            text = text[:cut + 1] if cut >= 0 else ""
        self._gen += 1
        for raw in text.splitlines():
            match = _LINE.match(raw)
            if not match:
                continue
            port, body = match.group(1), match.group(2)
            if port is None:
                self._router_line(body)
            else:
                self._child_line(int(port), body)
        self._flush(HOLD_CHUNKS)
        self._forget_stale()
        return self.requests, self.events

    # -- the router's own lines: which model is where, and swap events --------

    def _router_line(self, body):
        proxy = _PROXY.search(body)
        if proxy:
            # The one place the port a child listens on is tied to its alias.
            self.port_model[int(proxy.group(2))] = proxy.group(1)
            return
        loading = _LOADING.search(body)
        if loading:
            self.events.append({"event": "load", "model": loading.group(1)})
            return
        unload = _UNLOAD.search(body)
        if unload:
            self.events.append({"event": "unload", "model": unload.group(1)})
            return
        idle = _IDLE.search(body)
        if idle:
            # Not a swap in itself: the instance gave its slot up because
            # another model was queued behind it. Worth showing, because it is
            # why a model that was "loaded" a moment ago is suddenly not.
            self.events.append({"event": "idle", "model": idle.group(1)})

    # -- a model instance's lines: the actual requests ------------------------

    def _child_line(self, port, body):
        loaded = _LOAD_MODEL.search(body)
        if loaded:
            name = loaded.group(1).rsplit("/", 1)[-1]
            if name.endswith(".gguf"):
                name = name[:-5]
            self.port_model.setdefault(port, name)
            return
        task = _TASK.search(body)
        if not task:
            return
        key = (port, int(task.group(1)))

        if "launch_slot_" in body:
            # A slot runs one task at a time, so a new task starting on this
            # port proves the previous one is over and no `release` is coming
            # for it. This is the signal that actually closes a straddled
            # record; the age backstop below only catches the last one.
            self._flush_port(port, key[1])
            self.pending[key] = {"port": port, "task_id": key[1],
                                 "model": self.port_model.get(port)}
            return

        # Only a line that CARRIES token data may open a record. A poll can
        # land mid-request, so a timing line without its launch line is still
        # worth keeping -- but `total time` and `release` must never open one,
        # or the release that follows an already-emitted request opens a
        # second, empty record and emits it as a phantom call. Measured: 43
        # requests from a file that holds 35, only when read in small chunks.
        prompt = _PROMPT_EVAL.search(body)
        if prompt:
            record = self._track(key, port)
            record["prompt_ms"] = float(prompt.group(1))
            record["prompt_tokens"] = int(prompt.group(2))
            return
        evaluated = _EVAL.search(body.split("|")[-1])
        if evaluated:
            record = self._track(key, port)
            record["eval_ms"] = float(evaluated.group(1))
            record["generated_tokens"] = int(evaluated.group(2))
            return

        total = _TOTAL.search(body)
        if total:
            # `total time` is not the last line of a request: `release`
            # follows it with the context size and the truncation flag, so the
            # record moves to `done` and waits rather than being emitted here.
            record = self.pending.pop(key, None)
            if record is None:
                return
            record["total_ms"] = float(total.group(1))
            record["total_tokens"] = int(total.group(2))
            record["_gen"] = self._gen
            self.done[key] = record
            return

        release = _RELEASE.search(body)
        if release:
            record = self.done.pop(key, None) or self.pending.pop(key, None)
            if record is None:
                return
            record["context_tokens"] = int(release.group(1))
            record["truncated"] = int(release.group(2))
            self._emit(record)

    def _track(self, key, port):
        return self.pending.setdefault(
            key, {"port": port, "task_id": key[1],
                  "model": self.port_model.get(port)})

    def _flush(self, older_than=0):
        """Emit requests that saw `total time` but no `release`.

        `release` is two lines after `total time`, so a read boundary lands
        between them often enough to matter -- at 4 KB chunks it cost one
        record's context size in a real log, and at 211 bytes it cost most of
        them. A finished record is therefore held for a few chunks rather than
        emitted at the end of the one that completed it.
        """
        for key in sorted(self.done):
            if self._gen - self.done[key].get("_gen", 0) < older_than:
                continue
            self._emit(self.done.pop(key))

    def _flush_port(self, port, exclude_task):
        for key in sorted(self.done):
            if key[0] == port and key[1] != exclude_task:
                self._emit(self.done.pop(key))

    def _forget_stale(self):
        """Bound `pending`.

        A request whose `total time` never arrives -- a cancelled stream, a
        model evicted mid-generation -- leaves a partial record behind, and
        this parser lives for as long as the panel does. Measured: one router
        log left 35 of them. Oldest first, since tasks are issued in order.
        """
        limit = MAX_PENDING
        if len(self.pending) <= limit:
            return
        for key in sorted(self.pending)[:len(self.pending) - limit]:
            self.pending.pop(key, None)

    def _emit(self, record):
        """Finish one request record and add it to this chunk's output."""
        if not record:
            return
        record.pop("_gen", None)
        record.setdefault("model", self.port_model.get(record["port"]))
        record.setdefault("prompt_tokens", 0)
        record.setdefault("generated_tokens", 0)
        record.setdefault("prompt_ms", 0.0)
        record.setdefault("eval_ms", 0.0)
        record.setdefault("truncated", 0)
        # A request that ends without a `release` line has no context figure.
        # It must still be a number, not absent: the row is stored with
        # INSERT OR IGNORE, which drops a NOT NULL violation without a word.
        record.setdefault("context_tokens", 0)
        record["prompt_tok_s"] = _rate(record["prompt_tokens"], record["prompt_ms"])
        record["generated_tok_s"] = _rate(record["generated_tokens"], record["eval_ms"])
        self.requests.append(record)


def _rate(tokens, milliseconds):
    return round(tokens / (milliseconds / 1000.0), 2) if milliseconds else 0.0
