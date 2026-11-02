#!/usr/bin/env python3
"""Check llmlog.Parser against a real server log.

The panel has no test suite, and this is the one part of the AI tab where a
quiet bug produces plausible numbers instead of an error: a parser that drops
one request in thirty, or invents eight, still fills the tab.

The property that matters is **chunk invariance**. In production the parser
never sees a whole file -- it sees whatever bytes have landed since the last
poll, cut at an arbitrary byte. Every parser bug found here showed up only when
the input was chunked: a request lost to a split line, and phantom requests
conjured by a `release` line arriving after its record had been emitted.

    ./parse-test.py [/data/llm/logs/router-<jobid>.out]

Exits non-zero on a mismatch. Run it after any change to llmlog.py.
"""
import os
import sys
import glob

import llmlog

CHUNKS = (211, 997, 4096, 65536)


def newest_log():
    candidates = glob.glob("/data/llm/logs/*.out")
    if not candidates:
        sys.exit("no logs under /data/llm/logs -- pass one as an argument")
    return max(candidates, key=os.path.getmtime)


def key(record):
    return (record["port"], record["task_id"], record["prompt_tokens"],
            record["generated_tokens"], record.get("total_tokens"),
            record.get("context_tokens"), record["truncated"])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest_log()
    with open(path, errors="replace") as handle:
        text = handle.read()
    print("%s (%d bytes)" % (path, len(text)))

    whole = llmlog.Parser()
    reference, events = whole.feed(text)
    print("whole file: %d requests, %d model events, %d partial left"
          % (len(reference), len(events), len(whole.pending)))
    if not reference:
        print("WARNING: no completed requests in this log; the check is vacuous")

    expected = [key(r) for r in reference]
    failures = 0
    for size in CHUNKS:
        parser = llmlog.Parser()
        got = []
        for i in range(0, len(text), size):
            found, _ = parser.feed(text[i:i + size])
            got.extend(found)
        actual = [key(r) for r in got]
        if actual == expected:
            print("  %6d-byte chunks: %d requests, identical" % (size, len(got)))
            continue
        failures += 1
        print("  %6d-byte chunks: %d requests, MISMATCH" % (size, len(got)))
        missing = [k for k in expected if k not in actual]
        extra = [k for k in actual if k not in expected]
        for k in missing[:5]:
            print("      missing: %s" % (k,))
        for k in extra[:5]:
            print("      extra:   %s" % (k,))

    # A completed request must carry real numbers, not just exist.
    for record in reference:
        if record["prompt_tokens"] <= 0 and record["generated_tokens"] <= 0:
            print("  EMPTY record: %s" % (key(record),))
            failures += 1
            break

    if failures:
        sys.exit("FAILED: parsing depends on where the reads happen to be cut")
    print("OK: parsing is invariant to where the log is cut")


if __name__ == "__main__":
    main()
