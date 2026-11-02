"""Optional web lookup that enriches a finding with public threat-intel text.

This module exists under one hard rule: **nothing identifying the monitored
estate ever leaves the panel.** The alerts being analysed belong to a client
network, so a query is built ONLY from public vocabulary -- the Wazuh rule id,
the rule's own description text, and MITRE tactic names -- never from a
finding title, hostname, username, IP or any evidence alert. `_scrub` is a
second line of defence over that, not the first: the terms are whitelisted at
the source and then scrubbed again before they are sent.

Results are cached on disk by query, because rule ids repeat across findings
and a report of 50 findings usually asks fewer than a dozen distinct
questions.
"""
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request

import config

_lock = threading.Lock()
_ENDPOINT = "https://lite.duckduckgo.com/lite/"
_UA = "Mozilla/5.0 (compatible; slerm-panel/1.0)"

# Anything shaped like estate detail is removed even though it should never
# have been added: IPv4, IPv6-ish runs, UNC paths, emails, and DOMAIN\user.
_SCRUB = [
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    re.compile(r"\b[0-9a-f]{1,4}(?::[0-9a-f]{0,4}){2,}\b", re.I),
    re.compile(r"\\\\[^\s]+"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b[A-Za-z0-9_-]+\\[A-Za-z0-9_.$-]+"),
]
_RULE_IN_TEXT = re.compile(r"\(rule (\d+)\)")


def enabled():
    try:
        import wazuhllm
        return bool(wazuhllm.settings().get("web_search"))
    except Exception:  # noqa: BLE001
        return False


def _scrub(text):
    out = " ".join(str(text or "").split())
    for pattern in _SCRUB:
        out = pattern.sub(" ", out)
    return " ".join(out.split())[:160]


def terms_for(finding):
    """Public vocabulary describing what fired. Never estate detail.

    Returns [] when a finding has nothing safely describable, in which case no
    lookup happens at all -- an empty query is better than a leaky one.
    """
    base = finding.get("baseline") or {}
    parts = []
    for contributor in (base.get("contributors") or [])[:2]:
        desc = contributor.get("description") or ""
        parts.append(desc)
    if not parts:
        # rule descriptions are embedded in `why` as "... (rule NNNN)"
        why = finding.get("why") or ""
        for match in _RULE_IN_TEXT.finditer(why):
            parts.append("wazuh rule %s" % match.group(1))
            break
    rule = finding.get("rule")
    if rule and not parts:
        parts.append("wazuh rule %s" % rule)
    tactics = base.get("tactics") or []
    if tactics:
        parts.append("MITRE ATT&CK " + " ".join(tactics[:3]))
    if not parts:
        return []
    cleaned = [_scrub(p) for p in parts]
    return [c for c in cleaned if c]


def _cache_path():
    return os.path.join(getattr(config, "WAZUH_DIR", "/tmp"), "websearch.json")


def _load_cache():
    try:
        with open(_cache_path()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    path = _cache_path()
    tmp = path + ".part"
    try:
        with open(tmp, "w") as fh:
            json.dump(cache, fh)
        os.replace(tmp, path)
    except OSError:
        pass


# DuckDuckGo lite quotes its class attributes with SINGLE quotes; a
# pattern that only accepted double quotes matched nothing and every
# lookup silently returned zero results.
_RESULT = re.compile(
    r"<a[^>]+class=[\"']result-link[\"'][^>]*>(.*?)</a>.*?"
    r"<td[^>]*class=[\"']result-snippet[\"'][^>]*>(.*?)</td>",
    re.S | re.I)


_TAG = re.compile(r"<[^>]+>")


def _strip(html):
    text = _TAG.sub(" ", html or "")
    for entity, char in (("&amp;", "&"), ("&quot;", '"'), ("&#x27;", "'"),
                         ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " ")):
        text = text.replace(entity, char)
    return " ".join(text.split())


def search(query, timeout=15, limit=3):
    """Public search for one query string. Cached; never raises."""
    query = _scrub(query)
    if not query:
        return []
    with _lock:
        cache = _load_cache()
        hit = cache.get(query)
        if hit and time.time() - hit.get("at", 0) < 30 * 86400:
            return hit.get("results") or []
    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request(_ENDPOINT, data=data,
                                 headers={"User-Agent": _UA})
    results = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        for title, snippet in _RESULT.findall(body)[:limit]:
            results.append({"title": _strip(title)[:140],
                            "summary": _strip(snippet)[:320]})
    except Exception:  # noqa: BLE001
        # Enrichment is optional: a failed lookup must never fail a triage.
        return []
    with _lock:
        cache = _load_cache()
        cache[query] = {"at": time.time(), "results": results}
        _save_cache(cache)
    return results


def reference_for(finding):
    """Reference material for one finding, or [] when disabled or unsafe."""
    if not enabled():
        return []
    out = []
    for term in terms_for(finding)[:2]:
        for row in search(term):
            row = dict(row)
            row["query"] = term
            out.append(row)
        if len(out) >= 4:
            break
    return out[:4]
