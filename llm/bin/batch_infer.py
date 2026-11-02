#!/usr/bin/env python3
"""Push a file of prompts through a local llama-server and write JSONL results.

Loads the model once per job, unlike llama-cli, which reloads per prompt.
Sharded by SLURM_ARRAY_TASK_ID when run as an array, so N tasks split the file
N ways with no coordination.

Qwen3 is a reasoning model: with --jinja, llama.cpp splits the <think> block
out into reasoning_content and leaves content EMPTY if the token budget ran out
mid-thought. Both fields are captured below, and a too-small budget is flagged
rather than silently returning "".

Input:  JSONL with an "id" and a "prompt"/"messages" field, or plain text
        (one prompt per line).
Output: JSONL, one object per prompt, written incrementally so a killed job
        still leaves usable partial results.
"""
import json, os, sys, time, urllib.request, urllib.error, argparse


def load_items(path):
    items = []
    with open(path) as fh:
        for n, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if line[0] == "{":
                obj = json.loads(line)
                obj.setdefault("id", n)
                items.append(obj)
            else:
                items.append({"id": n, "prompt": line})
    return items


def post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--no-think", action="store_true",
                    help="disable Qwen3 reasoning (much cheaper per prompt)")
    args = ap.parse_args()

    items = load_items(args.infile)

    # Array sharding: task i handles every Nth item.
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    if tid is not None:
        count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
        items = items[int(tid)::count]
        print("shard %s/%s -> %d items" % (tid, count, len(items)), flush=True)

    done = 0
    with open(args.outfile, "w", buffering=1) as out:
        for it in items:
            if "messages" in it:
                msgs = it["messages"]
            else:
                msgs = []
                if args.system:
                    msgs.append({"role": "system", "content": args.system})
                msgs.append({"role": "user", "content": it["prompt"]})

            payload = {"messages": msgs,
                       "max_tokens": it.get("max_tokens", args.max_tokens),
                       "temperature": it.get("temperature", args.temperature)}
            if args.no_think:
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            t0 = time.time()
            try:
                r = post(args.url + "/v1/chat/completions", payload, args.timeout)
                choice = r["choices"][0]
                msg = choice["message"]
                rec = {"id": it["id"],
                       "response": msg.get("content") or "",
                       "usage": r.get("usage", {}),
                       "finish_reason": choice.get("finish_reason"),
                       "seconds": round(time.time() - t0, 2)}
                if msg.get("reasoning_content"):
                    rec["reasoning"] = msg["reasoning_content"]
                if not rec["response"]:
                    rec["warning"] = ("empty content - the token budget was spent "
                                      "on reasoning; raise --max-tokens or pass "
                                      "--no-think")
            except Exception as e:                      # keep going on one bad prompt
                rec = {"id": it["id"], "error": "%s: %s" % (type(e).__name__, e),
                       "seconds": round(time.time() - t0, 2)}
            out.write(json.dumps(rec) + "\n")
            done += 1
            print("[%d/%d] id=%s %.1fs%s" % (done, len(items), rec["id"],
                                             rec["seconds"],
                                             "  WARN" if rec.get("warning") else ""),
                  flush=True)
    print("wrote %s (%d records)" % (args.outfile, done))


if __name__ == "__main__":
    main()
