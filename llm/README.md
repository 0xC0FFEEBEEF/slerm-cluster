# LLM inference on slerm

llama.cpp on the two GPU nodes, served two ways: a persistent OpenAI-compatible
endpoint, and offline batch jobs. Everything lives under `/data/llm`, which is
NFS-shared to all nodes, so there is no deploy step — the share *is* the
install.

## Why llama.cpp and not vLLM

| | node03 | node04 |
|---|---|---|
| GPU | GTX 1660 SUPER (TU116) | GTX 980 Ti (GM200) |
| VRAM | 6 GB | 6 GB |
| Compute capability | 7.5 | **5.2** |
| Driver / max CUDA | 610.57 / 13.3 | 535.309 / **12.2** |

vLLM requires compute capability >= 7.0, so the 980 Ti cannot run it. llama.cpp
covers both cards from a single build. Two hard constraints follow:

- **Build with CUDA 12.x, never 13.** CUDA 13 dropped Maxwell support outright,
  and node04 driver 535 tops out at 12.2 anyway. The toolchain here is a
  userland CUDA 12.2 (no root was needed — see below).
- **`CMAKE_CUDA_ARCHITECTURES=52;75`** so one binary carries code for both.
- **`GGML_NATIVE=OFF`** with an explicit AVX2/FMA/F16C baseline. The build runs
  on a *compute* node and the binary executes on the *GPU* nodes; a native
  build would bake in whatever ISA the build host happened to have.

The 980 Ti is the slower of the two for quantized work: Maxwell predates the
`dp4a` int8 instruction that llama.cpp uses for Q4/Q8 matmuls, so it falls back
to a slower path. Prefer the Turing card for interactive use.

## Layout

    /data/llm/
      toolchain/cuda122/   userland CUDA 12.2 + cmake (micromamba, no root)
      src/llama.cpp/       upstream source
      build/bin/           llama-server, llama-cli, llama-bench, ...
      models/              GGUF weights
      jobs/                sbatch scripts
      bin/                 llm-env.sh, batch_infer.py
      run/                 endpoint-<node>.json while a server is up
      logs/                job output

`bin/llm-env.sh` is the single place that sets `PATH`/`LD_LIBRARY_PATH` and the
default model. Every job sources it.

## Rebuilding

    sbatch /data/llm/jobs/build.sh        # runs on compute (16 cores), ~15 min

The head has only 2 cores; building there is painful. Building on a compute node
is also why `GGML_NATIVE` has to stay off.

## Persistent endpoint

    sbatch --gres=gpu:turing:1  /data/llm/jobs/server.sh    # node03 (faster)
    sbatch --gres=gpu:maxwell:1 /data/llm/jobs/server.sh    # node04

Runs with `--time=INFINITE`, which the `gpu` partition allows. The API is then
at `http://slerm-node03:8080/v1` (or `.204`), OpenAI-compatible.

**The desktop can reach the GPU nodes directly; the head cannot.** That is the
opposite of what the panel assumes, and it is what makes a directly-reachable
endpoint possible at all. Point any OpenAI-style client at that base URL with
any dummy API key.

The server deliberately takes only **3 of 6 cores and 6G of 9.4G**, leaving room
for the NEXRAD array, which runs on `compute,gpu` without `--gres` to use these
nodes idle CPU cores. Do not raise this without checking that first.

## Batch inference

    sbatch /data/llm/jobs/batch.sh prompts.jsonl out.jsonl
    sbatch --array=0-1 /data/llm/jobs/batch.sh prompts.jsonl out.jsonl

The job starts its own llama-server on a private port, drains the file, and
exits — the card is free again afterwards. With `--array`, task *i* takes every
Nth prompt and writes `out.<i>.jsonl`, so both cards work the same file.

Input is JSONL (`{"id":..., "prompt":...}` or a `messages` array) or plain text,
one prompt per line. Output is JSONL written incrementally, so a killed job
still leaves usable partial results.

**Batch and endpoint coexist with no preemption config**: Slurm will not give
the same GPU to two jobs, so a batch job lands on whichever card is free. If you
want a batch job to *evict* a running server instead of queueing, see
`PREEMPTION.md` — that needs a root change to slurm.conf.

## Context size and VRAM

Qwen3-8B Q4_K_M is 4.7 GB against 6 GB of VRAM, so the KV cache budget is about
1 GB. The default is `LLM_CTX=4096`. To go bigger, quantize the cache rather
than spilling layers to CPU:

    LLM_CTX=8192 sbatch --gres=gpu:turing:1 /data/llm/jobs/server.sh

and add `--cache-type-k q8_0 --cache-type-v q8_0` to the `llama-server` line.
Watch `nvidia-smi` in the job log: once the model stops fully offloading, tokens
per second falls off a cliff.

## From the desktop

`~/.local/bin/slerm-llm` wraps all of it:

    slerm-llm status          # who is up, plus llm jobs in the queue
    slerm-llm up turing       # start a server
    slerm-llm chat "hello"    # streaming one-shot
    slerm-llm url             # base URL, for pointing other tools at
    slerm-llm batch in.jsonl out.jsonl
    slerm-llm down

## No sudo

`slerm` has no passwordless sudo on any of these hosts, so nothing here needed
root: the CUDA toolkit is a micromamba env on the NFS share, not an RPM. The one
thing that does need root is the optional preemption config in `PREEMPTION.md`.

## Qwen3 thinking mode — read this before trusting an empty response

Qwen3 is a reasoning model. With `--jinja`, llama.cpp splits the `<think>` block
into a separate `reasoning_content` field, so `content` comes back **empty** if
the token budget was spent before the model finished thinking. The first smoke
test returned `""` with `completion_tokens: 24` for exactly this reason.

`batch_infer.py` now captures both fields, records `finish_reason`, and attaches
an explicit `warning` instead of silently yielding `""`. Two ways out:

    LLM_INFER_ARGS=--no-think          sbatch ... jobs/batch.sh in.jsonl out.jsonl
    LLM_INFER_ARGS="--max-tokens 2048" sbatch ... jobs/batch.sh in.jsonl out.jsonl

`--no-think` sets `chat_template_kwargs.enable_thinking=false` and is much
cheaper per prompt. Against the live endpoint, the same switch is
`"chat_template_kwargs": {"enable_thinking": false}` in the request body.

## CPU fallback build

The CUDA build links `libcuda.so.1`, which exists **only on the GPU nodes** — it
cannot start on compute at all. `build-cpu/` is a CUDA-free build of the same
source for the 32 compute cores:

    sbatch /data/llm/jobs/build-cpu.sh      # ~2 min

    sbatch --partition=compute --gres=NONE --cpus-per-task=16 --mem=12G \
      --export=ALL,LLM_BUILD=/data/llm/build-cpu,LLM_NGL=0,LLM_CTX=2048 \
      /data/llm/jobs/batch.sh in.jsonl out.jsonl

`LLM_BUILD` in `bin/llm-env.sh` selects between the two; everything else is
identical. Useful when both cards are busy, or when the GPU nodes are down.
Expect roughly 2 s per short answer for Qwen3-8B Q4_K_M on 16 cores.

## The firewall (DONE 2026-08-31)

firewalld on the Rocky nodes blocked 8080. **Port 8080 is now open on node03 and
node04**, so the endpoint is directly reachable and `slerm-llm status` shows
`UP`. `slerm-llm` still keeps an SSH-tunnel fallback and will report `UP(ssh)`
if the direct route ever stops working (rule lost, firewalld reloaded from a
saved config), which makes that a useful signal rather than a hard failure.

For reference, this is what was run from the desktop:

    for h in slerm-node03 slerm-node04; do
      ssh -t $h "sudo firewall-cmd --permanent --add-port=8080/tcp && sudo firewall-cmd --reload"
    done

Tighter, if you would rather not expose it to the whole subnet -- replace
192.0.2.0/24 with the desktop address:

    ssh -t slerm-node03 "sudo firewall-cmd --permanent --add-rich-rule=\
    'rule family=ipv4 source address=192.0.2.0/24 port port=8080 protocol=tcp accept'\
     && sudo firewall-cmd --reload"

`slerm-llm` prefers the direct route and only tunnels when direct fails, so it
switches over on its own once the port is open.

## Measured throughput (Qwen3-8B Q4_K_M, ctx 4096, full offload)

| | node03 GTX 1660 SUPER (sm_75) | node04 GTX 980 Ti (sm_52) |
|---|---|---|
| generation | 45.0 tok/s | 28.6 tok/s |
| prompt | 82.9 tok/s | 27.1 tok/s |
| VRAM used | 5220 / 6144 MiB | 5439 / 6144 MiB |

The Maxwell card is ~1.6x slower generating and ~3x slower on prompt, which is
the missing `dp4a` int8 path. Send interactive work to node03.

## Host memory: why both GPU nodes could not run at once

The nodes are VMs on the Proxmox host `byteme`. A PCI-passthrough VM
must **pin its entire RAM** -- it cannot be ballooned or swapped. Starting the
second one demanded a hard 10 GiB the host did not have, so the kernel OOM
killer fired and took out the *already running* node, killing both:

    kvm invoked oom-killer ... vfio_pin_pages_remote
    oom-kill: task_memcg=/qemu.slice/303.scope
    303.scope: Failed with result oom-kill

Fixed 2026-08-31 by reclaiming ~6 GiB: `slerm-head` 8192 -> 4096 MB (it uses
813 MB) and ZFS `zfs_arc_max` 3.12 -> 1 GiB, both persistent. All three VMs now
run with ~6.5 GiB spare.

**Do not raise these back, and watch out starting other VMs on that host** --
there are 13 stopped VMs configured at >= 8000 MB each. Starting one now would
re-trigger the same OOM and kill a GPU node. The host has **no swap**: its btrfs
root is a 2-device RAID1 and btrfs refuses swapfiles on anything but a `single`
data profile, so headroom is the only protection.

## Context sizing on 6 GB — what was actually measured

The first Open WebUI chat failed with a misleading Open WebUI error,
`argument of type 'JSONResponse' is not iterable`. That is Open WebUI's error
*handler* crashing while formatting the real error, which was plain:

    srv send_error: request (5844 tokens) exceeds the available context size
    (4096 tokens), try increasing it

RAG retrieval plus chat history easily pushes a prompt past 4096. Measured on
node03 (GTX 1660 SUPER), Qwen3-8B Q4_K_M, full offload:

| ctx | KV | flash-attn | VRAM | gen tok/s | result |
|-----|-----|-----------|------|-----------|--------|
| 4096  | f16  | off | 5220 | 45.0 | fast, but rejects RAG prompts |
| 4096  | f16  | on  | 5220 | 36.2 | FA costs ~20% |
| 4096  | q8_0 | on  | -    | 31.4 | q8_0 costs a further ~13% |
| 8192  | f16  | on  | -    | -    | **OOM** (KV too big) |
| 8192  | f16  | off | -    | -    | **OOM** (compute buffer 564 MiB) |
| **8192** | **q8_0** | **on** | **5262** | **33.5** | **chosen default** |
| 12288 | q8_0 | on  | 5572 | 11.4 | fits, but only 572 MiB spare -> 3x slower |
| 16384 | q8_0 | on  | -    | -    | **OOM** |

Above ~8192 the card runs out of spare VRAM and throughput collapses; that
collapse is VRAM pressure, **not** context length in itself. 8192 with a q8_0 KV
cache is the sweet spot: it holds a 6221-token prompt at 35 tok/s with 882 MiB
still free. Flash attention is required for a quantised V cache, and on the
Maxwell card it also lifts prompt processing sharply (27 -> 67 tok/s).

Defaults now: `LLM_CTX=8192`, `LLM_CACHE=q8_0`, `LLM_FA=on`. Override per launch:

    sbatch --export=ALL,LLM_CTX=4096,LLM_CACHE=f16,LLM_FA=off --gres=gpu:turing:1 jobs/server.sh

## The TIME_WAIT bind trap

Restarting a server too quickly makes llama-server exit at once with
`couldn't bind HTTP server socket ... port: 8080`, even though nothing is
listening. llama-server does not set `SO_REUSEADDR`, so a socket lingering in
TIME_WAIT (~60s) blocks the bind. **`ss -l` will not show it** -- `-l` lists only
LISTEN. `server.sh` now waits on `ss -tanH` (all TCP states) before exec.

## Models migrated from cryoai (2026-09-01)

All 10 unique Ollama models copied into `/data/llm/models` (76 GB). Ollama stores
plain GGUF content-addressed, so **no conversion was needed** -- the blob behind
each manifest layer of mediaType `application/vnd.ollama.image.model` IS a GGUF
file, it just needs a real name. `qwen2.5-coder:7b` and `:latest` share one blob,
so 11 models = 10 files. **These are copies; nothing was deleted from cryoai.**

A GPU node has 6 GB VRAM (~5.2 GB usable after KV and compute buffers) plus
9.4 GB RAM; a compute node has 15.4 GB RAM and no GPU.

| model | size | full GPU offload | otherwise |
|---|---|---|---|
| llama3.2-1b       |  1.3 GB | yes | |
| qwen2.5-coder-3b  |  1.9 GB | yes | |
| qwen2.5-coder-7b  |  4.4 GB | yes | |
| deepseek-chat     |  4.6 GB | yes, tight | |
| qwen3-8b          |  4.9 GB | yes, ctx < 8192 | |
| qwen3.5-9b        |  6.6 GB | no | CPU, or RPC across both cards |
| qwen3-14b         |  8.7 GB | no | **RPC across both cards fits in VRAM** |
| qwen3-30b-a3b-iq2 |   11 GB | no | CPU (MoE, 3B active, tolerable) |
| gpt-oss-20b       |   13 GB | no | CPU only, very tight |
| qwen35-27b-cryoai |   17 GB | no | **fits nowhere** -- see below |

qwen35-27b-cryoai exceeds both a compute node RAM and the combined ~10.4 GB of
the two cards. Requantise to Q3_K_M (~13.5 GB) or Q2_K (~10 GB) with
`/data/llm/build/bin/llama-quantize` if you want it here.

## Pooling both GPUs with the RPC backend

`jobs/server-rpc.sh` runs one model across BOTH cards, so a model that fits on
neither alone runs entirely in VRAM.

    sbatch /data/llm/jobs/server-rpc.sh
    LLM_MODEL=/data/llm/models/qwen3-14b.gguf sbatch /data/llm/jobs/server-rpc.sh

**Verified working on qwen3-14b** (14.8B, 9.27 GB) at ctx 8192:

| | |
|---|---|
| node03 GTX 1660 SUPER | 5712 MiB |
| node04 GTX 980 Ti | 3866 MiB |
| total | 9.5 GB, fully offloaded, no CPU spill |
| throughput | 11.7 tok/s gen, 12.6 tok/s prompt |

### Two jobs, not one job with an srun step

`server-rpc.sh` runs on the turing node and **submits a separate
`rpc-worker.sh` job** for the maxwell card, then waits for port 50052.

It has to work this way. An `srun` step launched from node03 onto node04 **never
starts** -- it hangs and reports `aborted before step completely launched`,
whether or not `--gres` and `--overlap` are given. Step launch needs the worker
slurmd to connect BACK to `srun` on the origin node over ephemeral ports, and
firewalld here only permits 22/8080/50052 between the nodes. (`srun` from the
*head* works, which is why `slurmapi.gpu_probe` is fine.) Two independent jobs
talk solely over 50052, so none of that applies. Fixing it "properly" would mean
setting `SrunPortRange` in slurm.conf and opening that range -- a root change
that buys nothing here.

### The tensor-split ordering is backwards from what you would guess

`--tensor-split` maps positionally onto the device list, and **the RPC device
sorts FIRST, the local card second**. Measured: `60,40` put 5585 MiB on the
*remote* 980 Ti and only 4120 MiB on the local 1660 SUPER. So the default is now
**`40,60`** -- 40% remote maxwell, 60% local turing -- which is what favours the
faster card. Correcting it moved throughput from 10.9 to 11.7 tok/s.
`--list-devices` does NOT show RPC devices, so confirm by VRAM, not by that list.

Requires `-DGGML_RPC=ON` (enabled) and TCP 50052 open between the nodes. The
binary is named **`ggml-rpc-server`**, not `rpc-server`.

**Capacity, not speed.** Layer split is pipelined: the cards process a token in
sequence, not in parallel. The 8B on one card does 33.5 tok/s; the 14B across
both does 11.7. Use this only for models that do not fit on one card.

## GPU load bars in the panel, and iDRAC fan control

`bin/gpu-publish.py` runs inside every server job and writes nvidia-smi to
`/data/llm/run/gpu-<node>.json` every 5s. The panel reads it through
`slurmapi.gpu_live()` and draws util / vram bars on the node cards.

This sidesteps the panel long-standing constraint: it cannot ssh to the nodes,
and `slurmapi.gpu_probe` costs a real job slot so it can never be polled. A
server job is already sitting on the card, so sampling it is free. **Nothing
removes the file when a job dies** -- readers must treat anything older than 30s
as absent, which is what `GPU_LIVE_MAX_AGE` does. The probe button stays as the
fallback for an idle card.

### Fan control

`bin/gpu-fan-control.py` raises the chassis **minimum fan PWM** when the GPUs get
hot. Running as `systemctl --user gpu-fan-control` on the head.

**Why it is needed at all:** iDRAC cannot see the GPUs. Its sensors are Inlet,
Exhaust, CPU1 and CPU2 only, so the automatic curve is blind to the exact cards
that need cooling.

**One controller, not one per node.** Both GPU nodes are VMs on the same physical
host, so there is a single set of fans. Per-node control would have them fight,
the cooler one continually cancelling the hotter one. The controller runs once on
the head, reads the telemetry both nodes already publish, and acts on the hottest
card. No locking, because there is exactly one writer.

**MinimumFanSpeed, not FanSpeedOffset and not `ipmitool raw`.** It sets a floor
(10-100%); the automatic curve still runs on top and goes higher if the CPUs
demand it. We only ever raise the floor, never cap cooling, so if the controller
dies the fans revert to normal automatic behaviour. `255` means no floor.
The `ipmitool raw 0x30 0x30` route was rejected because it *disables* automatic
control outright -- and IPMI-over-LAN is disabled on this iDRAC anyway.

**Why racadm and not Redfish.** This is an iDRAC8 / 13G, firmware 2.83. Redfish
here is read-only for thermals: `/redfish/v1/Managers/iDRAC.Embedded.1/Attributes`,
which carries these settings on iDRAC9, returns **404**. racadm over SSH exposes
them. Auth is an SSH key installed with `racadm sshpkauth -i 2 -k 1 -t "<pubkey>"`,
so no password is stored. The key must be **RSA** -- iDRAC8 rejects ed25519.

Curve on the hottest GPU (quantised to 5% so it does not write constantly):

| GPU temp | minimum fan |
|---|---|
| < 60C | auto, no floor |
| 60C | 20% |
| 70C | 45% |
| 80C | 75% |
| >= 90C | 100% |

Raising the floor happens immediately; lowering it waits out a 90s dwell so the
fans do not surge and drop as a card flickers around a step.

Safety: exiting for any reason restores `255`, and stale or missing telemetry
does too -- unknown temps mean deferring to the iDRAC own thermal management,
which is always running underneath. `--apply` is required to write anything.

**Gotcha if you ever touch FanSpeedOffset instead:** `racadm get` returns a label
(`Low Fan Speed`) but `set` demands a numeric code, and the codes are NOT in
intensity order -- `0=Low, 1=High, 2=Medium, 3=Max, 255=Off`. Check
`racadm help system.thermalsettings.FanSpeedOffset` rather than guessing.

## Qwen3 thinking: disable it server-side, and mind the flag that does NOT work

Both server jobs now pass `--chat-template-kwargs '{"enable_thinking":false}'`
by default (`LLM_TKWARGS={}` restores thinking).

**`--reasoning off` does not disable Qwen3 thinking.** It looks like the right
flag and llama-server accepts it, but the model still emits a `<think>` block,
which lands in `message.reasoning_content` and leaves `message.content` EMPTY
when the token budget runs out. Only the template kwarg actually suppresses it.

This matters beyond wasted tokens: Open WebUI generates chat titles and tags by
asking the model for JSON and calling `JSON.parse` on the reply. With thinking
on, that reply is empty or prose, and the browser shows

    JSON.parse: unexpected character at line 1 column 1 of the JSON data

even though the chat request itself succeeded and the server logged HTTP 200.
The failing calls are the small background ones (~18 token prompts), not your
message -- check the llama-server log for short tasks right after a long one.

With the kwarg set, a title request returns `{"title": "Radar Signal Processing"}`
and parses cleanly.

## Open WebUI connection state while RPC is running

`server-rpc.sh` turns node04 into a bare RPC worker with no HTTP server, so
`http://slerm-node04:8080` goes dark. Open WebUI blocks its model list on
unreachable backends, so that connection is **disabled** in its config while RPC
runs (`api_configs["3"].enable = false`). Re-enable it when going back to two
single-card servers. The pooled model appears on the node03 connection as
`slerm-turing.qwen3-14b-rpc`.

The two modes are mutually exclusive -- one GPU can host one model, so it is
either the pooled 14B across both cards, or two independent single-card servers.

## Serving all the models at once: the router

`jobs/router-rpc.sh` is the RPC job in **router mode** -- one endpoint that
offers every model in `models/servable/` and loads them on demand across both
GPUs, instead of pinning a single model.

    sbatch /data/llm/jobs/router-rpc.sh

llama-server grew `--models-dir` / `--models-max` / `--models-preset`, so no
external proxy (llama-swap and friends) is needed. `LLM_MODELS_MAX` defaults to
**1**: with 6 GB cards only one model can be resident at a time, and the router
swaps on request. Measured switching between llama3.2-1b (35.7 tok/s) and
qwen3-14b (13.4 tok/s, pooled across both cards) works with no restart.

### Auto-fit is what makes oversized models work -- and two flags silently kill it

llama.cpp sizes each model to the free VRAM on its own, but **only when both
`--n-gpu-layers` and `--tensor-split` are left unset**. Setting either makes it
print

    common_fit_params: failed to fit params to free device memory:
      n_gpu_layers already set by user to 999, abort
      model_params::tensor_split already set by user, abort

and skip the fit entirely, after which anything larger than the pooled VRAM
simply OOMs. The router therefore sets **neither** by default. `LLM_NGL` and
`LLM_SPLIT` force them back, at the cost of only fully-fitting models loading.

That is what lets **qwen3-30b-a3b-iq2 (30B, 10.4 GB) run across the two 6 GB
cards at 11.8 tok/s** -- 4746 MiB on the 1660 SUPER, 5080 MiB on the 980 Ti. It
is MoE with ~3B active, which is why it is usable at all.

The single-model job `server-rpc.sh` keeps its tuned `40,60` split, because there
the model is known to fit and the tuning is worth ~7% throughput.

### What is deliberately not offered

`qwen35-27b-cryoai` (16.2 GB) exceeds even pooled VRAM plus sensible spill.
Requantise to Q3_K_M or Q2_K if you want it.

**The Ollama copy of gpt-oss cannot be served by llama.cpp at any VRAM budget:**

    error loading model: unknown model architecture: 'gptoss'

Ollama ships gpt-oss with its own architecture identifier (`gptoss`) because it
wrote a bespoke engine for it. The fix is the upstream conversion, whose arch is
`gpt-oss` with a hyphen: `ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf`,
downloaded here as `gpt-oss-20b-mxfp4.gguf` and **working across both cards**
(4648 + 5089 MiB, ~1.8 GB spilled to host RAM, 8.7 tok/s). MXFP4 runs fine on the
Maxwell card -- the CUDA backend was compiled for sm_52 and sm_75.

Do not bother requantising it: every quant is within 600 MB of MXFP4 (Q2_K is
still 10.9 GB) because the expert weights are natively MXFP4.

Every other transferred model uses a standard arch (`qwen3`, `qwen3moe`,
`qwen35`, `qwen2`, `llama`) and loads fine.

In Open WebUI they all appear on the existing node03 connection, prefixed
`slerm-turing.` -- no config change needed when adding a model, just drop a
symlink in `models/servable/` and restart the job.


## Thinking is ON, and what has to be true for that to work

Servers pass `--reasoning-format deepseek`, which routes thoughts into
`message.reasoning_content` -- Open WebUI renders that as a collapsible Thinking
block instead of leaking `<think>` into the reply. `LLM_TKWARGS` defaults to `{}`
so each model decides for itself; `'{"enable_thinking":false}'` forces it off.

**This only works because Open WebUI's task model is not a thinking model.**
Title, tag and follow-up generation ask the model for JSON and `JSON.parse` the
reply, which a `<think>` block breaks -- that is the
`JSON.parse: unexpected character at line 1 column 1` error. `task.model.default`
was pointing at `qwen2.5-0.5b-instruct`, **a model that does not exist here**, so
Open WebUI silently fell back to the *chat* model. It now points at
`llama3.2:1b` on cryoai's own Ollama, which also keeps it off the GPUs: the
router keeps only one model resident, so using a router model for tasks would
force a GPU model swap on every title.

Budget note: thinking eats tokens. Qwen3-8B spent ~3800 characters reasoning
about `17 * 23` and hit `finish_reason: length` at 1500 max_tokens. At 4000 it
finished cleanly. Leave max_tokens generous or thinking replies look empty.

## Spilling to compute-node CPUs over RPC -- and the trap that makes it pointless

`rpc-worker-cpu.sh` exposes a compute node CPU+RAM as an extra RPC device, so a
model too big for the two GPUs spills onto 16 cores and ~14 GB instead of the GPU
node own cramped 6 cores and ~3.4 GB.

    LLM_CPU_WORKER="slerm-node01,slerm-node02" sbatch /data/llm/jobs/router-rpc.sh

Note `LLM_CPU_WORKER` must be **exported**, not passed via `--export=ALL,VAR=a,b`
-- sbatch treats commas in `--export` as separators between variables, so a
comma-containing value silently truncates.

Needs `build-cpu` compiled with `-DGGML_RPC=ON` (the CUDA binary cannot run on a
node with no GPU) and TCP 50052 open on the compute nodes too.

### Auto-fit is HARMFUL once CPU devices are attached

Auto-fit allocates in proportion to **free memory**. A CPU node advertises ~14 GB
against ~5.7 GB per GPU, so it takes most of the layers and leaves the GPUs
nearly idle. Measured on gpt-oss-20b (11.3 GB):

| configuration | gen tok/s | GPU VRAM used |
|---|---|---|
| 2 GPUs, overflow to node03 local CPU | 8.72 | 4648 + 5089 MiB |
| + 2 CPU nodes, auto-fit | 8.76 | **1702 + 2052 MiB** |
| + 2 CPU nodes, split `45,5,5,45` | **22.94** | 5198 + 5543 MiB |

**2.6x**, purely from forcing the layers back onto the GPUs and letting the CPU
nodes absorb only the remainder. `router-rpc.sh` now derives a GPU-heavy split
automatically whenever CPU workers are attached (5% each to CPU, the rest split
between the two GPUs); `LLM_SPLIT` overrides it.

### Costs to be aware of

- **Small models get slower.** With CPU devices attached, llama3.2-1b dropped
  35.7 -> 19.6 tok/s. Run the plain GPU-only router for models under ~10 GB.
- **Loading takes ~4 minutes** across 4 devices instead of ~3, because weights
  ship over the network to each RPC worker.
- Latency is NOT the obstacle: node03->node01 is 0.606 ms against 0.548 ms for
  node03->node04 on the same hypervisor.

## More Ollama GGUFs that upstream llama.cpp cannot load

Alongside `gptoss`, both **qwen35** models from Ollama fail:

    error loading model hyperparameters:
      key qwen35.rope.dimension_sections has wrong array length; expected 4, got 3

The `qwen35` architecture IS supported by the build -- Ollama simply writes that
metadata field with 3 elements where upstream expects 4. Affects
`qwen3.5-9b` (6.6 GB, would otherwise fit easily) and `qwen35-27b-cryoai`.
Both removed from `models/servable/`. Lesson: **load-test every model before
adding it to the router**, since it will otherwise sit in the picker and fail
only when a user selects it.

## Can Ollama's qwen35 GGUFs be salvaged? Partly -- and the answer is instructive

`bin/fix-qwen35-rope.py` pads Ollama's 3-element rope sections to the 4 upstream
wants (`llama-hparams.h` declares `std::array<int,4> rope_sections` and
`llama-model.cpp` tests all four slots, so 0 is the correct pad, not a guess).

**It works, and it is not enough.** With the metadata fixed the array-length
error disappears and the next one appears:

    check_tensor_dims: tensor 'blk.0.ssm_dt.bias' not found

Qwen3.5 is a hybrid attention + state-space model, and Ollama lays those SSM
layers out differently from upstream. That is a structural difference in the
tensor set, not metadata, and patching tensor names would be guesswork.

**Conclusion: Ollama qwen35 GGUFs cannot be converted to upstream by editing.**
Use an upstream conversion (`unsloth/Qwen3.5-9B-GGUF`,
`bartowski/Qwen_Qwen3.5-9B-GGUF`), or for a local fine-tune re-run
`convert_hf_to_gguf.py` against the original safetensors. The patcher is kept
because the technique is sound for metadata-only mismatches.

Two gotchas the script had to handle, worth remembering for any GGUF rewriting:
- `GGUFReader.fields` includes **synthetic header entries** (`GGUF.version`,
  `GGUF.tensor_count`, `GGUF.kv_count`). Writing those back out as metadata
  yields a file whose header appears twice and no reader will open it.
- An **empty array** field carries no element type and cannot be serialised.
  This model has an empty `qwen35.vision.deepstack_visual_indexes`; it is
  meaningless for a text model and is dropped.

## 30B MoE across the pooled devices -- measured

Both `Qwen3-30B-A3B-Instruct-2507` quants run on two 6 GB cards plus the two
compute-node CPU devices:

| model | size | tok/s |
|---|---|---|
| qwen3-30b-a3b-2507-q3km | 13.7 GiB | 15.02 |
| qwen3-30b-a3b-2507-q4km | 17.3 GiB | 13.87 |

Q4 costs only ~8% over Q3, so Q4 is the better default. ~3B active parameters is
what makes CPU spill cheap enough for this to work at all.

### One split cannot serve mixed model sizes

`LLM_SPLIT=30,20,20,30` is sized for the 17.3 GiB Q4: 30% of it is ~5.2 GB,
which fits each GPU, leaving ~3.5 GB per CPU node. The auto-derived
`45,5,5,45` (tuned for gpt-oss at 11 GB) would demand **7.8 GB on a 6 GB card**
and OOM.

The cost is that gpt-oss then gets only ~3.4 GB per GPU instead of the 5.2 GB
that gave it 22.94 tok/s. **Pick the split for the largest model you intend to
load, or run one model at a time via `server-rpc.sh` with a split tuned to it.**
`--models-preset` (an INI of per-model settings) is the unexplored option that
might allow per-model tensor-split without a custom router.

## Per-model settings with `--models-preset` (the real fix for mixed sizes)

`/data/llm/etc/models.ini` gives every model its own `tensor-split` and
`ctx-size`, which is what a single global split could never do. Format is INI:
`[*]` for globals, one section per model name, keys are command-line arguments
without the leading dashes.

**A share of `0` excludes a device.** That is the important discovery: small
models get `tensor-split = 50,0,0,50` and never touch the CPU nodes at all.

| model | split | tok/s |
|---|---|---|
| llama3.2-1b | `50,0,0,50` | **69.04** (was 19.6 with CPU nodes attached) |
| gpt-oss-20b-mxfp4 | `45,5,5,45` | 21.90 |
| qwen3-30b-a3b-2507-q4km | `26,24,24,26` @ 32k ctx | 12.63 |

GPU shares are picked so `share x model_size` is about 5 GB, which fits a 6 GB
card alongside its KV slice.

**CLI arguments outrank the preset**, so the router must NOT pass `--ctx-size`
or `--tensor-split` on the command line when a preset is in use -- doing so
silently overrides every per-model value. `router-rpc.sh` drops those flags and
makes the auto-derived split stand down whenever `models.ini` exists.

Also note a preset section advertises a model **even when the file is missing**,
so it will appear in the picker and fail on load. Comment out sections for
models that are still downloading.

### Context size is nearly free

    ctx  8192 / split 30,20,20,30    13.87 tok/s
    ctx 32768 / split 26,24,24,26    12.63 tok/s

4x the context for ~9%, and part of that is the split shift rather than the
context. Qwen3-30B-A3B trains at 262144, so the ceiling here is VRAM for KV --
which is why a few points moved from the GPUs to the CPU nodes.

### Open WebUI has no auto-compaction

There is no native setting to summarise or trim long chats. The available levers
are server-side context (now per-model in the preset) and per-model `max_tokens`
in the UI advanced params. True compaction would need a custom Open WebUI
Function/Filter.

## Two Tesla P100s replace the 1660 SUPER + 980 Ti (2026-10-04)

Both P100-PCIE-16GB (GP100, sm_60, passive) sit in slerm-node04 (VM 304: 12 vCPU, 16 GB;
82:00 = CUDA0 at Gen3 x16, 04:00 = CUDA1 at Gen3 x8 because that root port is x8). node03 is
retired. Driver 535.309 (node03's 610 has no Pascal support; 580 is the last branch with it).
Build: `build-pascal/` (`jobs/build-pascal.sh`, CUDA 12.2, arch 60).

### One card, llama-bench (pp512 / tg128, fa on, all layers on the card)

    llama3.2-1b Q8_0          4099 / 178    (old pool ~70 gen)
    qwen3-8b Q4_K_M            655 /  38    (1660 SUPER 33.5, 980 Ti 22-24)
    ornith-1.5-9b Q4_K_M       592 /  33.5  (old pool 104 / 25.8)
    qwen3-14b Q4_K_M           218 /  21.7  (old pool 123 / 15-16.6)
    gpt-oss-20b MXFP4          292 /  61.6  (old pool 85 / 18-23)
    qwen3-30b-a3b Q3_K_M       410 /  52.0  (old pool 15.0 with 2 CPU nodes)
    gpt-oss-20b @ 32k depth    304 /  59.4

GP100 has no dp4a (only GP102+ do), so quantized mat-vec is emulated and generation reaches
~1/4 of the 732 GB/s. No throttling at any point: 1328 MHz, <= 200 W of the 250 W cap.

### Long context

gpt-oss-20b at ctx 65536 on one card (12.3 GB): real prompts of 20k / 45k / 60k tokens ran at
393 / 339 / 311 tok/s prompt and 47 / 37.5 / 32.6 tok/s generation. The old pool's ~30.2k
"ceiling" (two Xid 31 crashes) was the 980 Ti.

### Through the gateway (presets as shipped, ~2k-token prompt, 450-token reply for gen)

    model                      cards  load   prompt  gen     VRAM GPU0/GPU1 (MiB)
    gpt-oss-20b (64k)          1      161 s  210     45.0    12538 / -
    qwen3-14b (32k, q8 KV)     1       60 s  275     19.6*   11968 / -
    qwen3-30b-a3b Q4 (64k)     2      253 s  321     39.2    11422 / 10860
    qwen3-coder-30b Q4 (64k)   2      254 s  356     44.4*   11422 / 10860
    gemma4-26b-a4b Q6 (32k)    2      359 s  291     23.9     12292 / 11804
    holo4-35b-a3b (32k+mmproj) 2      315 s  167     36.1     12118 / 10492
    coder-next Q2 (32k)        2      375 s  202     20.4     13848 / 13140
    bonsai-27b Q1_0            1       49 s  131     17.7      6016 / -
    (* short-reply figure; the others are 450-token replies. Short replies under-report gen
     badly -- bonsai read 1.2 and gemma4 2.2 tok/s on a 2-token answer.)

All 22 shipped models load with their full configured context and answer. Load times are
NFS-bound (~60-90 MB/s) until the node-local cache holds a copy.

### Cooling: iDRAC alone is enough

iDRAC lists the cards but has no sensor for them, yet ramps the fans with their temperature.
With `gpu-fan-control` stopped (now disabled) and 15 min of sustained prompt processing:
one card plateaued at 72 C (peak 73, fans ~13.4k RPM); both cards at 65-67 C (peak 68, fans
~11.4-12.2k RPM, byteme 574 W). Slowdown is 82 C and SHUTDOWN 85 C, so re-test when the room
is warm (inlet was 21 C). Idle fan baseline rose from ~3.2k to ~8.5-9.6k RPM with the cards in.

### Bigger models (2026-10-04, through the gateway, 450-token replies)

    model                               where              load    prompt  gen
    qwen3.6-35b-a3b UD-Q4_K_XL +mmproj  2 cards            ~285 s  213-280 44-50 (load-mode none)
    ornith-1.5-35b-a3b Q4_K_M           2 cards            261 s   360     36.2
    kat-coder-v2.5-dev Q4_K_M           2 cards            141 s*  378     46.6*
    nemotron-3.5-lightning-30b-a3b      2 cards            294 s   317     48.5
    qwen3-coder-next UD-Q4_K_XL         2 cards + node05   18 min  20      7.3
    qwen3.6-35b-a3b UD-Q6_K_XL          2 cards + node05   449 s   52      9.9
    gpt-oss-120b MXFP4                  2 cards + n01/02/05 25 min 13.7    4.9
    (* with load-mode none; 248 s / 34.3 tok/s before it)

Spill costs far more than the share suggests: 14% of qwen3.6 Q6 on deadlock cut generation from
~44 to 9.9 and prompt from ~280 to 52. gpt-oss-120b pushed byteme2 to 4.9 of 8 GB swap.

### mmap + the cache keeper = stalls

model-cache.py (the router's keeper) copies recently used GGUFs to node04's local disk from INSIDE the
router job (cgroup --mem=10G). With the default mmap load, that copy's page cache evicted the served
model's mmapped CPU-side tensors, so every token waited on a re-read: qwen3.6-35b fell from 44 to
12.5-14.6 tok/s with both GPUs at 0-44% util and full clocks. `load-mode = none` in models.ini [*] fixed
it: 50.3 / 44.0 / 43.8 tok/s measured while the keeper copied kat-coder (20 GB).

### Qwen3.8-27B (added 2026-10-04 evening)

Dense 27.8B, qwen35 architecture (hybrid attention; a 65th MTP block loads fine on this build), vision. unsloth
UD-Q4_K_XL 16.4 GiB + F16 mmproj on both cards at ctx 131072: 12.2 + 12.4 GB VRAM. Through the gateway: 11.8 tok/s
generation (450-token reply), 150 tok/s prompt, image prompt 57 tok/s; answers correct. About a quarter of
Qwen3.6-35B-A3B's speed (the MoE reads ~3B per token, this reads all 27.8B), for a large jump on Qwen's card
(OSWorld-Verified 84.3 vs 63.9 for Qwen3.6-27B, SWE-bench Pro 61.7 vs 53.5).

### Ornith-1.5-35B-A3B speed bench (2026-10-06)

Question: any speedup for Ornith without losing quality? Bench in `bench/ornith-20261006/` (`bench.sh` /
`bench2.sh` + `client*.py`: one llama-server per config on node04, production args, 128k ctx + mmproj,
Ornith's coding sampler; `kld.sh` = quality check). Workloads: short code (1-1.5k tokens), a full-file edit
(opencode's pattern), 29.6k and 90k-token prompts with 600-1500 tokens generated at that depth.

    config (all ctx 131072)              short gen  edit   gen@29.6k  gen@90k  prompt@29.6k  prompt@90k
    production (layer, q8_0 KV, ub512)   47.7       45.4   38.1       26.8     527           411
    + MTP draft-mtp n-max 3              39.1       40.7   34.8       -        454           -
    + MTP + ngram-mod                    38.8       42.9   34.6       -        450           -
    -sm tensor                           42.5       41.4   45.7       -        517           -
    f16 KV                               49.6       48.6   46.4       ~41*     537           -
    ub/b 2048                            48.0       45.3   38.6       -        799           -
    layer + f16 KV + ub2048              49.8       48.9   -          41.3     -             571
    tensor + f16 KV + ub2048             38.9       43.1   -          37.7     -             567
    (* gen speed does not depend on ub; measured 41.3 with ub2048)

- **f16 KV is the win, and it is the precise format.** Pascal pays for dequantising q8_0 KV inside the
  flash-attention kernel, which is why Ornith fell from 48 to 27 tok/s as context grew. Only 10 of 40 layers
  keep KV (full attention every 4th), so 128k f16 is ~0.4 GB more: 13.7 / 12.0 GB.
- **MTP is a net loss on stock kernels.** Ornith's GGUF carries a trained MTP head (`nextn_predict_layers=1`)
  and b10729 loads it, but acceptance is 30-40% (mean 1.9-2.2 tokens per step) while a step costs ~2.3x a
  token: every draft runs the 248k-vocab LM head, and verifying 4 tokens makes each MoE layer read up to 4x
  the experts. It also slows prompts (the MTP layer processes them too). The community P100 patches 07/08
  target exactly this batched-MoE path; untested here.
- **`-sm tensor` loses once KV is f16**: no P2P between the cards (PHB in the VM), so the per-layer sync goes
  through the host. Its win at 29.6k on q8_0 KV was the dequant cost, which f16 removes for both modes.
  Layer split keeps each card ~55% busy during generation by design (they take turns per token).
- **ngram-mod**: +4.5% on the edit (drafts ~31 tokens long when it hits), neutral at 90k. Measured on tensor only.
- **ub 2048: +39-52% prompt speed, but not bit-neutral.** On sm_60 bigger batches move more expert matmuls
  from the int8 MMVQ path to cuBLAS FP16. `llama-perplexity --kl-divergence`, 2 x 8192 tokens, reference =
  layer + f16 KV + ub512: production (q8_0 KV) KLD 0.0145, same top token 97.4%, PPL 2.331; f16 + ub2048
  KLD 0.0182, 97.1%, PPL 2.320 (reference 2.336). Same order as the q8_0 KV drift, PPL slightly lower; left
  as the user's decision. GPU0 reaches 15.5 GB with ub2048 + f16 at 128k.
- Clocks are not a lever: the cards boost to 1328 MHz under load and never throttled.
- Applied 2026-10-06: `cache-type-k/v = f16` in `[ornith-1.5-35b-a3b]` (backup
  `etc/models.ini.bak-ornith-f16kv-20261006`).
- Live check through the gateway after deploy (router 4453): short 49.5 tok/s; 90k prompt 419 tok/s, then
  41.2 tok/s generated at that depth (was 26.8).

### Precision follow-up: FP32 accumulation (2026-10-06 evening, applied)

llama.cpp runs cuBLAS with CUBLAS_COMPUTE_16F on "fast fp16" cards, which includes the P100, so prompt processing
accumulates in FP16 (`ggml_cuda_mul_mat_cublas`; override `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`). Generation is not
affected (batch-1 MMVQ accumulates in FP32; the FA vec kernel keeps KQ max/sum and VKQ in float on NVIDIA).
Reference = FP32 accumulation + f16 KV + ub512, `llama-perplexity --kl-divergence -c 8192 --chunks 4`
(`bench/ornith-20261006/prec.sh`):

    config                         KLD      same top   PPL (ref 2.0006)
    q8_0 KV, ub512 (old prod)      0.0102   97.79%     1.9995
    f16 KV, ub512 (10-06 noon)     0.0084   97.85%     2.0011
    f16 KV, ub2048, FP16 accum     0.0106   97.72%     1.9938
    f16 KV, ub2048, FP32 accum     0.0070   98.18%     1.9970   <- applied

Server, 128k ctx: f32 + ub2048 read a 90k prompt at 624 tok/s (FP16 ub2048 571, ub512 419) and generated 40.5 tok/s
at that depth (41.2), 45.7 short (49.5). VRAM 15.5 / 13.9 GB. Applied: `export GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` in
`jobs/router-rpc.sh` (router-wide; backup `.bak-f32-20261006`) and `batch-size`/`ubatch-size = 2048` for Ornith
(`etc/models.ini.bak-ornith-ub2048-20261006`). Lower PPL than the reference is noise, not quality.
