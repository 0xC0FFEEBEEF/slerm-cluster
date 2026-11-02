"""Optional LLM triage for Wazuh findings (LM Studio, OpenAI-compatible).

This layer is ADVISORY and strictly additive. It never edits, reorders,
rescores or filters a finding: the detectors are deterministic and a report
must stay reproducible, so triage is written to a SEPARATE sidecar
(<report>.triage.json) keyed by finding index. Delete the sidecar and the
report is exactly what the cluster computed.

The model is asked for three things per finding: what the alert actually
means, what would have triggered it, and whether it looks like a false
positive. That last one is a suggestion for a human, never an auto-dismiss --
nothing in the pipeline drops a finding because a model said so.
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

import config

_lock = threading.Lock()

SYSTEM = (
    "You are a SOC analyst assistant reviewing alerts from a Wazuh SIEM. "
    "For each finding you are given the detector that produced it, the "
    "statistical baseline it was judged against, and up to three raw alerts "
    "as evidence, and where available the host's own history and the other "
    "findings that name it, and sometimes public reference material about the "
    "rule. Reference material is background only and describes the wider "
    "world, never this network: never cite it as evidence that something "
    "happened here. Use that history: volume that is ordinary for a "
    "busy host is not evidence of anything. Explain what the alert means and "
    "what activity would have "
    "triggered it, then judge whether it looks genuine or like routine noise. "
    "Be concrete and brief. If the evidence does not support a confident "
    "judgement, say so rather than guessing. Reply with STRICT JSON only, no "
    "markdown fences, matching exactly: "
    '{"explanation": str, "trigger": str, "assessment": '
    '"likely_genuine"|"needs_review"|"likely_false_positive", '
    '"confidence": "low"|"medium"|"high", "reasoning": str}'
)

VALID_ASSESSMENTS = ("likely_genuine", "needs_review", "likely_false_positive")

# Tunables the UI may change, with the bounds a saved value must satisfy.
# url and token are deliberately absent: they are infrastructure and a secret,
# they live in llm.env, and neither is ever sent to a browser.
_FIELDS = {
    "model":       ("str",   None),
    "max_tokens":  ("int",   (128, 32768)),
    "timeout":     ("int",   (10, 3600)),
    "temperature": ("float", (0.0, 2.0)),
    "top_n":       ("int",   (1, 200)),
    "json_schema": ("bool",  None),
    "thinking":    ("bool",  None),
    "web_search":  ("bool",  None),
    "all_findings": ("bool", None),
    "auto":        ("bool",  None),
}
_DEFAULT_KEYS = {
    "model": "WAZUH_LLM_MODEL", "max_tokens": "WAZUH_LLM_MAX_TOKENS",
    "timeout": "WAZUH_LLM_TIMEOUT", "temperature": "WAZUH_LLM_TEMPERATURE",
    "top_n": "WAZUH_LLM_TOP_N", "json_schema": "WAZUH_LLM_JSON_SCHEMA",
    "thinking": "WAZUH_LLM_THINKING", "web_search": "WAZUH_LLM_WEB_SEARCH",
    "all_findings": "WAZUH_LLM_ALL_FINDINGS",
    "auto": "WAZUH_LLM_AUTO",
}
_settings_cache = {"mtime": None, "data": {}}


def settings_path():
    return getattr(config, "WAZUH_LLM_SETTINGS_PATH", "")


def _overrides():
    """Saved overrides, re-read whenever the file changes."""
    path = settings_path()
    if not path:
        return {}
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        _settings_cache["mtime"] = None
        _settings_cache["data"] = {}
        return {}
    if _settings_cache["mtime"] != mtime:
        try:
            with open(path) as fh:
                loaded = json.load(fh)
            _settings_cache["data"] = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            _settings_cache["data"] = {}
        _settings_cache["mtime"] = mtime
    return _settings_cache["data"]


def coerce(field, value):
    """Validate one field, raising LLMError with a message a user can act on."""
    kind, bounds = _FIELDS[field]
    if kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in ("0", "false", "no", "off", "")
    if kind == "str":
        text = ("" if value is None else str(value)).strip()
        if len(text) > 200:
            raise LLMError("%s is too long" % field)
        return text
    try:
        number = int(value) if kind == "int" else float(value)
    except (TypeError, ValueError):
        raise LLMError("%s must be a number" % field)
    low, high = bounds
    if not (low <= number <= high):
        raise LLMError("%s must be between %s and %s" % (field, low, high))
    return number


def settings():
    """Effective settings: saved overrides layered over the config defaults."""
    saved = _overrides()
    out = {}
    for field, cfg_key in _DEFAULT_KEYS.items():
        default = getattr(config, cfg_key, None)
        if field in saved:
            try:
                out[field] = coerce(field, saved[field])
                continue
            except LLMError:
                pass          # a bad saved value must never break triage
        out[field] = default
    if out.get("temperature") is None:
        out["temperature"] = 0.1
    return out


def save_settings(values):
    """Persist validated overrides. Only known fields are written."""
    path = settings_path()
    if not path:
        raise LLMError("no settings path configured")
    clean = {}
    for field, value in (values or {}).items():
        if field not in _FIELDS:
            continue          # url/token are not settable from a browser
        clean[field] = coerce(field, value)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, 0o700)
    tmp = path + ".part"
    with open(tmp, "w") as fh:
        json.dump(clean, fh, indent=1, sort_keys=True)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    _settings_cache["mtime"] = None
    return settings()


def describe():
    """Everything the settings UI needs: current values, bounds, and models."""
    probe = health()
    return {"settings": settings(),
            "defaults": {f: getattr(config, k, None)
                         for f, k in _DEFAULT_KEYS.items()},
            "overridden": sorted(_overrides().keys()),
            "fields": {f: {"kind": k, "min": b[0] if b else None,
                           "max": b[1] if b else None}
                       for f, (k, b) in _FIELDS.items()},
            "models": probe.get("models") or [],
            "url": probe.get("url") or getattr(config, "WAZUH_LLM_URL", ""),
            "source": probe.get("source"),
            "node": probe.get("node"), "job_id": probe.get("job_id"),
            "loaded": probe.get("loaded"), "warning": probe.get("warning"),
            "ok": probe.get("ok"), "state": probe.get("state"),
            "error": probe.get("error"),
            "path": settings_path()}


# Grammar-constrained output. This is not a nicety: prism-ml/bonsai-27b is a
# reasoning model that, left unconstrained, streams an unbounded monologue and
# never closes it -- it burned 1995 of 2000 tokens thinking and returned an
# empty answer on every finding. Constraining generation to this schema makes
# it terminate in ~250 tokens (31s instead of 250s+), and the answer is valid
# JSON by construction rather than by hope.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "trigger": {"type": "string"},
        "assessment": {"type": "string", "enum": list(VALID_ASSESSMENTS)},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"},
    },
    "required": ["explanation", "trigger", "assessment", "confidence"],
}


class LLMError(RuntimeError):
    pass


# --- where the model actually runs -------------------------------------------
# The cluster router cannot be reached by a fixed URL: its Slurm job moves to
# whichever GPU node it is given on every requeue, so a pinned host is stale as
# soon as it is requeued once. llmusage already discovers it from the queue for
# the AI tab, so triage resolves through that rather than keeping a second copy
# of the same logic. The pinned WAZUH_LLM_URL remains for anything else that
# speaks the OpenAI API (LM Studio, Ollama) and is what runs with
# WAZUH_LLM_CLUSTER off.

def endpoint():
    """Where this call should go, and how that was decided.

    Never raises: callers get {"url": None, "error": <why>} instead, because
    "no server is running right now" is an ordinary state on a cluster, not a
    misconfiguration.
    """
    if not getattr(config, "WAZUH_LLM_CLUSTER", False):
        url = (getattr(config, "WAZUH_LLM_URL", "") or "").rstrip("/")
        return {"url": url or None, "source": "static", "server": None,
                "error": None if url else "no WAZUH_LLM_URL configured"}
    try:
        import llmusage
        found = llmusage.servers()
    except Exception as exc:                                    # noqa: BLE001
        return {"url": None, "source": "cluster", "server": None,
                "error": "could not read the Slurm queue: %s" % exc}
    up = [s for s in found if s.get("ok") and s.get("url")]
    if not up:
        broken = [s for s in found if s.get("url")]
        why = ("%s is not answering (%s)"
               % (broken[0].get("node") or broken[0].get("name"),
                  broken[0].get("error") or "no reason given")) if broken else (
              "no inference server is running on the cluster -- submit one "
              "from /data/llm/jobs")
        return {"url": None, "source": "cluster", "server": None, "error": why}
    # Prefer a server that already holds the wanted model. The router runs
    # --models-max 1 and NFS feeds it at ~57 MB/s, so a swap is minutes of
    # blocked call, and it evicts whatever else was using the card.
    want = settings().get("model")
    # 2026-10-17: triage talks to the router directly, and asking it for a model
    # that is not resident swaps the model -- which would kick a locked one off.
    lock = llmusage.current_lock()
    if lock and want and want != lock.get("model"):
        return {"url": None, "source": "cluster", "server": None,
                "error": "slerm is locked to %s; triage uses %s and waits until it is unlocked"
                         % (lock.get("model"), want)}
    chosen = None
    for server in up:
        if want and server.get("loaded") == want:
            chosen = server
            break
    if chosen is None:
        chosen = up[0]
    return {"url": chosen["url"].rstrip("/") + "/v1", "source": "cluster",
            "server": chosen, "error": None}


def _auth_headers(where):
    # The cluster router has no auth, and the token in llm.env belongs to LM
    # Studio. Sending it there would be meaningless at best.
    headers = {}
    if where.get("source") == "static":
        token = getattr(config, "WAZUH_LLM_TOKEN", None)
        if token:
            headers["Authorization"] = "Bearer " + token
    return headers


def configured():
    """Whether triage can run at all right now.

    In cluster mode this is a live question, not a settings one -- the answer
    changes when the router job is cancelled -- but llmusage.servers() is
    cached for a few seconds, so asking is cheap.
    """
    return bool(endpoint().get("url"))


def _post(path, body, timeout=None):
    where = endpoint()
    if not where.get("url"):
        raise LLMError(where.get("error") or "no LLM endpoint available")
    url = where["url"] + path
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(_auth_headers(where))
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(
                req, timeout=timeout or settings()["timeout"]) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:200]
        except Exception:  # noqa: BLE001
            pass
        raise LLMError("%s HTTP %d: %s" % (where["url"], exc.code, detail))
    except urllib.error.URLError as exc:
        raise LLMError("unreachable at %s (%s)" % (where["url"], exc.reason))
    except (ValueError, OSError) as exc:
        raise LLMError(str(exc))


def health():
    where = endpoint()
    if not where.get("url"):
        state = "disabled" if where["source"] == "static" else "no server"
        return {"ok": False, "state": state, "error": where.get("error"),
                "source": where["source"], "url": None}
    server = where.get("server") or {}
    base = {"source": where["source"], "url": where["url"],
            "node": server.get("node"), "job_id": server.get("job_id"),
            "loaded": server.get("loaded")}
    try:
        req = urllib.request.Request(where["url"] + "/models",
                                     headers=_auth_headers(where))
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id") for m in (data.get("data") or [])]
        model = (_overrides().get("model") or config.WAZUH_LLM_MODEL
                 or (models[0] if models else None))
        out = dict(base, ok=True, state="ok", models=models, model=model)
        # Worth saying out loud: with --models-max 1 a model that is not the
        # resident one costs a full load off NFS before the first answer, and
        # evicts whatever was on the card.
        if where["source"] == "cluster" and model and server.get("loaded") \
                and model != server["loaded"]:
            out["warning"] = ("%s is resident; asking for %s will swap the "
                              "model, which takes minutes"
                              % (server["loaded"], model))
        elif where["source"] == "cluster" and model and not server.get("loaded"):
            out["warning"] = ("nothing is resident; the first call loads %s, "
                              "which takes minutes" % model)
        if model and models and model not in models:
            out["ok"] = False
            out["state"] = "model missing"
            out["error"] = "%s is not served here; pick one of the %d listed" \
                           % (model, len(models))
        return out
    except urllib.error.HTTPError as exc:
        state = "unauthorised" if exc.code in (401, 403) else "error"
        return dict(base, ok=False, state=state,
                    error="HTTP %d from %s" % (exc.code, where["url"]))
    except Exception as exc:  # noqa: BLE001
        return dict(base, ok=False, state="unreachable", error=str(exc))


def _shorten(alert, limit=700):
    """Evidence lines are whole alerts; trim so the prompt stays bounded."""
    try:
        parsed = json.loads(alert) if isinstance(alert, str) else alert
    except ValueError:
        return str(alert)[:limit]
    keep = {}
    for path in ("timestamp", "@timestamp", "rule", "agent", "full_log",
                 "data", "location"):
        if path in parsed:
            keep[path] = parsed[path]
    text = json.dumps(keep, separators=(",", ":"))
    return text[:limit]


def _prompt(finding, context=None):
    evidence = [_shorten(e) for e in (finding.get("evidence") or [])[:3]]
    payload = {
        "detector": finding.get("detector"),
        "title": finding.get("title"),
        "why_it_fired": finding.get("why"),
        "statistical_baseline": finding.get("baseline"),
        "already_demoted_as_routine": finding.get("routine"),
        "demotion_reason": finding.get("routine_reason"),
        "evidence_alerts": evidence,
    }
    if finding.get("reference_material"):
        # Public background on what the rule means. Explicitly labelled as
        # unverified third-party text so the model does not treat a search
        # snippet as evidence about THIS estate.
        payload["reference_material_unverified"] = finding["reference_material"]
    if context:
        # Volume alone means little without knowing what this host normally
        # does: a domain controller carrying 4% of the estate is not unusual
        # for a domain controller. The sibling findings matter too -- one host
        # appearing under several detectors is a different claim from one.
        payload["context"] = context
    return json.dumps(payload, separators=(",", ":"))


def _json_candidates(text):
    """Yield balanced {...} blocks, last first.

    A greedy \{.*\} spans from the first brace in the model's REASONING to the
    last brace of the answer and produces garbage. Bonsai is a reasoning model,
    so that happened intermittently -- the same finding parsed on one run and
    failed on the next. Scanning for balanced braces and preferring the LAST
    block picks the answer rather than the thinking.
    """
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in reversed(starts):
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break


def _parse(text):
    """LM Studio may wrap JSON in prose or fences however firmly we ask."""
    text = (text or "").strip()
    # reasoning models emit their thinking inline; drop it before looking for JSON
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    parsed = None
    try:
        parsed = json.loads(text)
    except ValueError:
        for candidate in _json_candidates(text):
            try:
                got = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(got, dict) and ("assessment" in got or
                                          "explanation" in got):
                parsed = got
                break
        if parsed is None:
            raise LLMError("model did not return JSON (%d chars: %r)"
                           % (len(text), text[:120]))
    if not isinstance(parsed, dict):
        raise LLMError("model returned %s, not an object" % type(parsed).__name__)
    assessment = str(parsed.get("assessment") or "needs_review").lower().strip()
    if assessment not in VALID_ASSESSMENTS:
        # never silently accept an invented verdict
        assessment = "needs_review"
    confidence = str(parsed.get("confidence") or "low").lower().strip()
    if confidence not in ("low", "medium", "high"):
        confidence = "low"
    return {
        "explanation": str(parsed.get("explanation") or "")[:1200],
        "trigger": str(parsed.get("trigger") or "")[:800],
        "assessment": assessment,
        "confidence": confidence,
        "reasoning": str(parsed.get("reasoning") or "")[:800],
    }


# LM Studio evicts and reloads a model on its own (JIT loading, idle TTL, a
# competing request for a different model). That surfaces as an HTTP 400
# "Model reloaded." against a request that was otherwise fine, so it is worth
# one more attempt. A timeout is NOT in this list: it means the box is too slow
# for the configured budget, and retrying only burns the same wall clock again.
_TRANSIENT = ("model reloaded", "model_not_found", "is loading",
              "loading the model", "http 503", "http 429")


def _attempt(finding, context=None):
    cfg = settings()
    body = {
        "model": cfg["model"] or "local-model",
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": _prompt(finding, context)}],
        "temperature": cfg["temperature"],   # stable, not creative
        "max_tokens": cfg["max_tokens"],
        "stream": False,
    }
    if not cfg["thinking"]:
        # Ask the chat template to skip the reasoning turn. Qwen3-family and
        # Nemotron templates honour enable_thinking; some do not -- Bonsai
        # ignored both this and reasoning_effort and reasoned anyway -- so
        # this is a request, not a guarantee. Both spellings are sent because
        # LM Studio passes chat_template_kwargs straight to the template and
        # different templates read different names.
        body["chat_template_kwargs"] = {"enable_thinking": False,
                                        "thinking": False}
        body["reasoning_effort"] = "low"
    if cfg["json_schema"]:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "wazuh_triage", "strict": True,
                            "schema": RESPONSE_SCHEMA},
        }
    data = _post("/chat/completions", body)
    choices = data.get("choices") or []
    if not choices:
        raise LLMError("model returned no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if not (content or "").strip():
        # LM Studio routes a reasoning model's whole turn into
        # reasoning_content and leaves content empty -- including the
        # schema-constrained answer. The reply was there all along; we were
        # reading the wrong field and calling it "unavailable".
        content = message.get("reasoning_content") or content
    truncated = choice.get("finish_reason") == "length"
    if truncated and not (content or "").strip():
        # A reasoning model can burn the whole budget thinking and return
        # nothing at all. Say that plainly rather than blaming the JSON.
        usage = (data.get("usage") or {}).get("completion_tokens_details") or {}
        raise LLMError(
            "model hit the %d-token limit while reasoning (%s reasoning tokens) "
            "and returned no answer -- raise the max_tokens setting"
            % (cfg["max_tokens"], usage.get("reasoning_tokens", "?")))
    try:
        return _parse(content)
    except LLMError:
        # The answer started and was cut off mid-object. That is the same
        # budget problem as above, not a model that cannot follow the schema,
        # and reporting it as "did not return JSON" sent us chasing the prompt.
        if truncated:
            raise LLMError(
                "model hit the %d-token limit and its JSON answer was cut off "
                "mid-object -- raise the max_tokens setting"
                % cfg["max_tokens"])
        raise


def triage_one(finding, attempts=2, context=None):
    if not configured():
        raise LLMError("LLM triage is not configured")
    last = None
    for i in range(attempts):
        try:
            return _attempt(finding, context)
        except LLMError as exc:
            last = exc
            lowered = str(exc).lower()
            if not any(m in lowered for m in _TRANSIENT):
                raise
            if i + 1 < attempts:
                time.sleep(3)
    raise last
