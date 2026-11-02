"""Wazuh indexer client -- profiles, health, and paged alert fetch.

Uses urllib rather than requests: the panel venv has neither requests nor
numpy, and adding a dependency for four HTTP calls is not worth it.

Credentials live in a mode-600 file OUTSIDE this directory (see
config.WAZUH_PROFILES_PATH) and never reach the browser -- the profile API
returns urls and usernames, never a password. The panel itself has no
authentication and is served to the whole 192.0.2.0/24 subnet, so the users
configured here are expected to be read-only and scoped to wazuh-alerts-*.
"""
import base64
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import threading
import urllib.request

import config


class WazuhError(RuntimeError):
    pass


# Never shrink below this: at some point the fetch is simply too slow to be
# useful and the real answer is a narrower window.
MIN_PAGE = 100
# How many times one page may be retried before the fetch really fails.
OVERLOAD_RETRIES = 6

_OVERLOAD_MARKERS = (
    "circuit_breaking_exception", "data too large",
    "task_cancelled_exception", "too_many_requests",
    "es_rejected_execution", "search_context_missing",
)


# Concurrency gate. The parent circuit breaker is a property of TOTAL heap in
# flight, not of any one request: fetch #8 tripped it with 1.1kb reserved
# because 996mb was already committed. Halving a page cannot fix pressure that
# is cumulative, so concurrency itself has to give way -- and permanently for
# the life of the fetch, since a cluster that just refused load will refuse it
# again a second later.
_gate = None
_gate_lock = threading.Lock()
_gate_size = 0


def _gate_init(size):
    global _gate, _gate_size
    with _gate_lock:
        if _gate is None:
            _gate_size = max(1, int(size))
            _gate = threading.Semaphore(_gate_size)
    return _gate


def _gate_shrink():
    """Retire one permit, never below one. Returns the new ceiling."""
    global _gate_size
    with _gate_lock:
        if _gate_size <= 1:
            return 1
        _gate_size -= 1
        target = _gate_size
    # consume a permit without ever giving it back
    threading.Thread(target=_gate.acquire, daemon=True).start()
    return target


def _is_overload(exc):
    """Is this the indexer refusing load rather than refusing the request?"""
    text = str(exc).lower()
    return any(marker in text for marker in _OVERLOAD_MARKERS)


# A blip on the wire, as distinct from a tunnel that is genuinely down. Fetch
# 12 crossed a 39ms WAN link for 27 minutes and 3,013,092 alerts, then lost
# every one of them to a single TLS handshake timeout, because a URLError was
# fatal and only overload markers were retried. Over a transfer that long a
# transient error is not an edge case, it is expected at least once.
_TRANSIENT_NET_MARKERS = (
    "timed out", "timeout", "handshake", "connection reset",
    "broken pipe", "eof occurred", "connection aborted",
    "temporary failure in name resolution", "no route to host",
)
NET_RETRIES = 5


def _is_transient_net(exc):
    """A retryable transport failure, not a refusal and not a dead tunnel.

    Deliberately narrow: an unreachable profile must still surface as
    unreachable rather than as an empty window, so this only survives a
    bounded number of blips before the original error is raised.
    """
    text = str(exc).lower()
    if "rejected the credentials" in text or "returned http" in text:
        return False
    return any(marker in text for marker in _TRANSIENT_NET_MARKERS)


_profiles_cache = {"mtime": None, "data": {}}


def load_profiles():
    """Read the credentials file, cached on mtime so edits are picked up."""
    path = config.WAZUH_PROFILES_PATH
    try:
        stat = os.stat(path)
    except OSError:
        _profiles_cache.update(mtime=None, data={})
        return {}
    if _profiles_cache["mtime"] == stat.st_mtime:
        return _profiles_cache["data"]
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise WazuhError("cannot read %s: %s" % (path, exc))
    if not isinstance(data, dict):
        raise WazuhError("%s must be a JSON object of profiles" % path)
    _profiles_cache.update(mtime=stat.st_mtime, data=data)
    return data


def profile(name):
    profiles = load_profiles()
    if name not in profiles:
        raise WazuhError("unknown Wazuh profile %r" % name)
    raw = profiles[name]
    if not isinstance(raw, dict):
        raise WazuhError("Wazuh profile %r must be a JSON object" % name)
    prof = dict(raw, id=name)
    if not prof.get("url"):
        raise WazuhError("Wazuh profile %r has no url" % name)
    return prof


def profile_ids():
    return sorted(load_profiles().keys())


def _context(prof):
    if prof.get("verify") is False or prof.get("verify") is None:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    if prof.get("verify") is True:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=prof["verify"])


def request(prof, path, body=None, method=None, timeout=None):
    """One call against the indexer. Raises WazuhError with a useful message."""
    url = prof["url"].rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if prof.get("user"):
        token = "%s:%s" % (prof["user"], prof.get("password", ""))
        headers["Authorization"] = "Basic " + base64.b64encode(
            token.encode("utf-8")).decode("ascii")

    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(
                req, timeout=timeout or config.WAZUH_TIMEOUT,
                context=_context(prof)) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        if exc.code in (401, 403):
            raise WazuhError("%s rejected the credentials (HTTP %d)"
                             % (prof["id"], exc.code))
        raise WazuhError("%s returned HTTP %d: %s" % (prof["id"], exc.code, detail))
    except urllib.error.URLError as exc:
        # This is the tunnel-is-down case, and it must not be reported as
        # "no alerts" -- an empty result and an unreachable indexer look
        # identical in the UI otherwise.
        raise WazuhError("%s unreachable at %s (%s)"
                         % (prof["id"], prof["url"], exc.reason))
    except (ValueError, OSError) as exc:
        raise WazuhError("%s: %s" % (prof["id"], exc))


def health(name):
    """Probe one profile. Never raises -- the UI renders the failure."""
    started = time.time()
    try:
        prof = profile(name)
    except WazuhError as exc:
        return {"profile": name, "ok": False, "state": "unconfigured",
                "error": str(exc)}
    try:
        info = request(prof, "/", timeout=config.WAZUH_HEALTH_TIMEOUT)
    except WazuhError as exc:
        message = str(exc)
        state = "unreachable"
        if "credentials" in message:
            state = "unauthorised"
        return {"profile": name, "ok": False, "state": state,
                "url": prof.get("url"), "user": prof.get("user"),
                "error": message, "elapsed": round(time.time() - started, 3)}
    version = (info.get("version") or {}).get("number")
    return {"profile": name, "ok": True, "state": "ok",
            "url": prof.get("url"), "user": prof.get("user"),
            "cluster": info.get("cluster_name"), "version": version,
            "elapsed": round(time.time() - started, 3)}


def indices(name, pattern=None):
    prof = profile(name)
    pattern = pattern or prof.get("index") or config.WAZUH_INDEX_PATTERN
    # Keep wildcards useful, but do not let a configured pattern introduce a
    # second path or query string into this request.
    encoded = urllib.parse.quote(str(pattern), safe="*,-_.")
    rows = request(prof, "/_cat/indices/%s?format=json&h=index,docs.count"
                   % encoded)
    out = []
    for row in rows or []:
        out.append({"index": row.get("index"),
                    "docs": int(row.get("docs.count") or 0)})
    return sorted(out, key=lambda r: r["index"], reverse=True)


def _query(start, end, extra=None, end_exclusive=False):
    # Adjacent slices share a boundary instant. With lte on both sides an alert
    # landing exactly on it is fetched by BOTH neighbours; every slice but the
    # last therefore uses lt.
    upper = "lt" if end_exclusive else "lte"
    must = [{"range": {"@timestamp": {"gte": start, upper: end,
                                      "format": "strict_date_optional_time"}}}]
    if extra:
        must.append({"query_string": {"query": extra}})
    return {"bool": {"must": must}}


def fetch(name, start, end, pattern=None, filter_query=None, max_alerts=None,
          page=None, on_batch=None, end_exclusive=False, should_stop=None):
    """Page the whole window out of the indexer, oldest first.

    `search_after` on a [@timestamp, _id] sort rather than scroll: it holds no
    server-side context, so a slow consumer cannot expire mid-fetch. The sort
    includes _id to break ties, without which alerts sharing a millisecond are
    silently skipped at a page boundary.
    """
    prof = profile(name)
    pattern = pattern or prof.get("index") or config.WAZUH_INDEX_PATTERN
    try:
        page = int(config.WAZUH_PAGE_SIZE if page is None else page)
        limit = int(config.WAZUH_MAX_ALERTS
                    if max_alerts is None else max_alerts)
    except (TypeError, ValueError):
        raise WazuhError("page size and alert cap must be integers")
    if page <= 0 or limit <= 0:
        raise WazuhError("page size and alert cap must be positive")
    encoded = urllib.parse.quote(str(pattern), safe="*,-_.")
    body = {
        # This is adjusted on every iteration so a cap smaller than a page (or
        # a final partial page) never leaks extra alerts to the callback.
        "size": min(page, limit),
        "query": _query(start, end, filter_query, end_exclusive),
        "sort": [{"@timestamp": "asc"}, {"_id": "asc"}],
        "_source": {"excludes": list(config.WAZUH_SOURCE_EXCLUDES)},
        # Bounded, not True. An exact total makes the indexer count every
        # matching document across every shard and reduce it in one buffer,
        # which is what tripped the parent circuit breaker at ~974mb. We
        # only ever display this as "about N", so a ceiling is enough.
        "track_total_hits": config.WAZUH_TRACK_TOTAL_HITS,
    }
    after = None
    total = 0
    reported = None
    reported_exact = True
    good = 0
    page_max = int(getattr(config, "WAZUH_PAGE_MAX", page))
    gate = _gate_init(getattr(config, "WAZUH_MAX_INFLIGHT", 2))
    while True:
        if should_stop is not None and should_stop():
            break
        remaining = limit - total
        if remaining <= 0:
            break
        body["size"] = min(page, remaining)
        if after:
            body["search_after"] = after
        result = None
        attempts = 0
        while result is None:
            gate.acquire()
            try:
                result = request(prof, "/%s/_search" % encoded, body=body)
            except WazuhError as exc:
                attempts += 1
                # The parent circuit breaker is a function of PAGE SIZE, not of
                # the window: a smaller page reduces into a smaller buffer and
                # the identical query then succeeds. Halve and retry rather
                # than failing a fetch that is merely asking for too much at
                # once. Sequential search_after paging means a smaller page
                # costs round trips, never correctness.
                if _is_transient_net(exc) and not _is_overload(exc):
                    if attempts > NET_RETRIES:
                        raise
                    # A blip is not a capacity problem: keep the page size and
                    # the concurrency, just wait for the wire to come back.
                    delay = min(30.0, 2.0 ** attempts)
                    sys.stderr.write(
                        "wazuh: transport blip (%s), retry %d in %.0fs\n"
                        % (str(exc)[:80], attempts, delay))
                    waited = 0.0
                    while waited < delay:
                        if should_stop is not None and should_stop():
                            raise
                        time.sleep(min(1.0, delay - waited))
                        waited += 1.0
                    continue
                if not _is_overload(exc) or attempts > OVERLOAD_RETRIES:
                    raise
                # back off on BOTH axes: fewer requests in flight, and smaller
                # ones. Sleep grows so the indexer gets time to GC rather than
                # being asked again while it is still over the limit.
                ceiling = _gate_shrink()
                if body["size"] > MIN_PAGE:
                    page = max(MIN_PAGE, body["size"] // 2)
                    body["size"] = min(page, remaining)
                good = 0
                delay = min(30.0, 2.0 ** attempts)
                sys.stderr.write(
                    "wazuh: indexer overloaded, retry %d in %.0fs "
                    "(page=%d, concurrency=%d)\n"
                    % (attempts, delay, body["size"], ceiling))
                # Sleep in slices so a cancellation lands in about a second
                # rather than after a full 30s backoff.
                waited = 0.0
                while waited < delay:
                    if should_stop is not None and should_stop():
                        raise
                    time.sleep(min(1.0, delay - waited))
                    waited += 1.0
            finally:
                gate.release()
        good += 1
        if good >= 3 and page < page_max:
            page = min(page_max, page * 2)   # ramp back up on a healthy cluster
            good = 0
        hit_block = result.get("hits") or {}
        hits = hit_block.get("hits") or []
        if reported is None:
            found = hit_block.get("total")
            if isinstance(found, dict):
                reported = found.get("value")
                # bounded track_total_hits reports a FLOOR once the ceiling is
                # hit ("gte"), not a total -- reporting it as one produced
                # "143,500 of 50,000 alerts fetched"
                reported_exact = found.get("relation", "eq") == "eq"
            else:
                reported = found
        if not hits:
            break
        # A conforming indexer honours size. Defensively trim anyway: the
        # persisted snapshot must obey WAZUH_MAX_ALERTS exactly.
        if len(hits) > remaining:
            hits = hits[:remaining]
        if on_batch:
            on_batch([h.get("_source") or {} for h in hits], total, reported,
                     reported_exact)
        total += len(hits)
        if total >= limit:
            break
        after = hits[-1].get("sort")
        if not after:
            break
    return {"fetched": total, "reported": reported,
            "reported_exact": reported_exact, "index": pattern,
            "capped": bool(reported is not None and reported_exact
                           and total >= limit and total < reported)}


def slice_window(start, end, slices):
    """Cut [start, end] into `slices` contiguous half-open ranges.

    Parsed as epoch millis so the split is arithmetic rather than calendar
    arithmetic; returns ISO strings the indexer accepts. The final slice keeps
    an inclusive upper bound so the caller's own `end` is never lost.
    """
    import datetime as _dt

    def _parse(value):
        text = str(value).strip().replace("Z", "+00:00")
        stamp = _dt.datetime.fromisoformat(text)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=_dt.timezone.utc)
        return stamp

    begin, finish = _parse(start), _parse(end)
    if slices < 2 or finish <= begin:
        return [(start, end, False)]
    span = (finish - begin) / slices
    out = []
    for i in range(slices):
        left = begin + span * i
        right = finish if i == slices - 1 else begin + span * (i + 1)
        out.append((left.isoformat(), right.isoformat(), i != slices - 1))
    return out
