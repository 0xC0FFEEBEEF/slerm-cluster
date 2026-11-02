# deadlock / slerm-node05 — provisioning + LLM benchmark results

**Session:** 2026-09-09 evening → 2026-09-10 overnight
**Node:** `deadlock` = Slurm `slerm-node05`
**Status:** node in production, benchmarks running overnight

---

## TL;DR

- deadlock is live in `compute` + `llmcpu`, survived a reboot, and is serving as a 4th RPC device for llama.cpp.
- **gpt-oss-20b got 6.4% faster** with deadlock in the pool (24.64 → 26.22 tok/s).
- Getting it into the RPC pool needed **a new llama.cpp build** — see "The BMI2 trap".
- Results for the three large models are appended at the bottom as they finish.

---

## The hardware

| | value |
|---|---|
| CPU | Xeon E5-1620 v2 — 4 cores / 8 threads @ 3.7 GHz (**Ivy Bridge**) |
| RAM | **31,756 MiB** — 2× node01/02, 3× the GPU nodes |
| GPU | none (the Matrox G200e is BMC video) |
| Disk | 2.7 TB (70 G root, 15.7 G swap, **2.64 T unused LV**) |
| ISA | `avx`, `f16c` — **no avx2, no fma, no bmi2** |

It is a **RAM node, not a compute node**: it holds more layers than either compute
node but computes each one more slowly.

---

## Four things that bit, and why

### 1. The RTC was 91 days fast
It read **Dec 9** on Sep 9. A clock in the *future* makes valid mirror certs read as
**expired**, so `dnf` failed with a confusing TLS error. It would also have broken
munge silently later (munge rejects credentials across clock skew).
**Fix:** time sync now runs *before* package installs in the provisioner.

### 2. `/home` was a local 2.6 TB LVM volume
The installer carved one, so a naive `mountpoint -q /home` check skipped the NFS
mount and would have left the node with a private home — job scripts silently
wrong rather than visibly broken.
**Fix:** local entry commented out in `/etc/fstab`, NFS `/home` mounted over it.
The LV `rlm_deadlock-home` (2.64 T) is **preserved and unmounted**. To reclaim:
```
mount /dev/mapper/rlm_deadlock-home /scratch
```

### 3. EPEL ships Slurm 22.05.9
It fights the cluster's local 25.05.8 RPMs. Installing anything beyond
`slurm` + `slurm-slurmd` drags in EPEL's `slurm-perlapi` and the transaction fails
on file conflicts.
**Fix:** `exclude=slurm*` now set on **all six hosts** (head + node01-05).

### 4. The BMI2 trap (the expensive one)
`build-cpu/` is compiled to a Haswell baseline. Every binary in it dies instantly
on deadlock:
```
exit 132, SIGILL, core dumped
```
Rebuilding with `-DGGML_AVX2=OFF -DGGML_FMA=OFF` **still SIGILLed**. Tracing the
faulting address through the ELF program headers found it:
```
1bf60:  c4 e2 79 f7 f7    shlx %eax,%edi,%esi     <- BMI2, inside ggml_cpu_init
```
`GGML_BMI2` defaults to the same `${INS_ENB}` as AVX2 but is a **separate option**,
and because the offending instruction sits in `ggml_cpu_init` itself, *every*
binary died at startup — even `llama-cli --version`.

**Fix:** `-DGGML_BMI2=OFF` → `/data/llm/build-cpu-avx/`.

### 5. `slerm-node05` had no DNS record
The router resolves workers with `getent hosts "$W"` using the **Slurm** name.
Slurm learned `slerm-node05` from `NodeHostname=deadlock`, but DNS never did, so
`CIP` came back empty, the probe became `/dev/tcp//50052`, failed its full 120 s,
and the router then **cancelled the worker job**. Cost two router restarts.
**Fix:** A record for `slerm-node05` in Technitium.
(PTR deliberately left pointing at `deadlock` — that is the real hostname.)

---

## Benchmark results

Method: `POST /v1/chat/completions`, fixed prompt, `max_tokens=200`,
`temperature=0`. First request is a warmup (excluded) because `MODELS_MAX=1` means
the router swaps models on demand and run 1 carries the load. Figures are the
median of the timed runs, using llama.cpp's own `predicted_per_second`.

### Test 1 — gpt-oss-20b-mxfp4 (11.3 GiB, MoE)

| pool | tok/s | prompt tok/s | model load |
|---|---|---|---|
| 4 devices (node04, node01, node02, node03) | 24.64 | 31.9 | 10 s |
| **5 devices (+deadlock)** | **26.22** | 33.1 | 135 s |

**+6.4%.** Both GPU shares held at 45% in each case; only the CPU spill changed
(5+5 → 3+3+4), so the comparison isolates the added device.

I predicted this would be *slower* — inserting an AVX2-less device into a
pipelined split ought to add the slowest link. What that missed: the CPU share
stayed at 10% while spreading thinner, so no single CPU device sits on the
critical path as long, and deadlock's 3.7 GHz clock offsets its missing AVX2 at
that small a share. The real cost is **model load: 10 s → 135 s.**

### Tests 2-4 — appended below as they complete

---

## What changed on the cluster

**New files**
- `/data/llm/build-cpu-avx/` — AVX-only llama.cpp (no avx2/fma/bmi2)
- `/data/llm/jobs/build-cpu-avx.sh` — builds it
- `/data/llm/jobs/rpc-worker-cpu-avx.sh` — node05's RPC worker (8 threads, 28 G)
- `/data/llm/bin/gguf-meta3.py` — dependency-free GGUF metadata reader
- `~/slerm-provision-node.sh` — reusable 11-step node provisioner
- `~/bin/slerm-llm-ctl` — scoped router control (see below)
- `~/.config/technitium/api-token` — DNS API creds (0600)

**Modified**
- `/data/llm/jobs/router-rpc.sh` — picks the CPU worker build per node, so node05
  gets the AVX build instead of SIGILLing
- `/data/llm/etc/models.ini` — 5-device splits; three models added
- `/etc/slurm/slurm.conf` — now a **symlink to `/apps/slurm.conf` on all 6 hosts**,
  so a node change is one edit + `scontrol reconfigure` instead of six pushes
- systemd drop-ins: `RequiresMountsFor=/apps` on slurmd/slurmctld/slurmdbd

**Backups:** `models.ini.bak-*`, `slurm.conf.bak-*`, `epel.repo.bak-*`,
`/etc/fstab.bak-*`, `router-rpc.sh.bak-*`

---

## Control tool

```
slerm-llm-ctl status                      # jobs + endpoint + rpc devices
slerm-llm-ctl restart [worker-list]       # cancel + resubmit the router
slerm-llm-ctl worker-mem <node> <mem>     # resubmit a CPU worker with new --mem
slerm-llm-ctl models                      # what the router is serving
slerm-llm-ctl logs [n]                    # tail the live router log
```

Every `scancel` resolves the job id **by name** from `squeue`, so it cannot cancel
anything on the cluster that is not an llm job.

---

## Disk expansion — DONE 2026-09-10

`/export` grew from 400 G to **700 G (351 G free, was 53 G)**.

Worth recording because the obvious command failed: **`growpart` is not installed
on the head** (it ships in `cloud-utils-growpart`, absent from Rocky's base
install). Running `xfs_growfs /export` on its own then *looked* like it worked —
it printed the existing geometry (`blocks=104857088`) and exited 0 — but changed
nothing, because the partition was still 400 G. Easy to misread as success.

The sequence that worked:
```
sudo dnf install -y cloud-utils-growpart
sudo growpart /dev/sdb 1
sudo xfs_growfs /export
```

Result: `/dev/sdb1  700G  350G  351G  50% /export`

## Still worth doing sometime (cosmetic)

node01-04 keep their pre-symlink local config copies at
`/etc/slurm/slurm.conf.localcopy-*`. Harmless, just clutter.

**Also worth doing sometime:** node01-04 still keep their pre-symlink local config
copies at `/etc/slurm/slurm.conf.localcopy-*`. Harmless, just clutter.

### Test 2 — qwen3-coder-next-ud-q4kxl (46.2 GiB, MoE) — **THE CAPACITY PROOF**

This model **cannot run at all** without deadlock.

| | |
|---|---|
| pool without deadlock | 10.4 VRAM + 14 + 14 = **38.4 GiB** → does not fit |
| pool with deadlock | 10.4 + 14 + 14 + 28 = **66.4 GiB** → fits |
| split | `10,21,21,38,10` |
| **generation** | **5.47 tok/s** (median of 2; wall-clock agrees at 5.20) |
| prompt | 2.41 tok/s |
| **model load** | **1114 s (18.6 min)** — 46 GiB across 1 GbE NFS + RPC |

Measured placement during load, matching the split almost exactly:

| device | held | of |
|---|---|---|
| node01 | 11 G | 15 G |
| node02 | 10 G | 15 G |
| **deadlock** | **17-18 G** | **31 G** |

deadlock's ~18 G is the share no other node in the cluster could have taken.

**Context:** the q2kxl quant (24.9 GiB) was previously the largest model that ran
here, at 7.09 tok/s. So the **higher-quality q4kxl runs at 77% of the q2kxl's
speed** while being 1.9x the size. That trade — better weights for ~23% less
throughput — is what the RAM node bought.

MoE is why this is usable at all: 10 of 512 experts active, ~3B params per token,
so most of those 46 GiB sit idle on any given token and the CPU nodes only have to
move the active slice.

**Caveat:** an 18.6-minute load makes this a "park it and use it" model, not one to
put behind a router that swaps on demand (`MODELS_MAX=1`). Pin it if you want it.

### Test 3 — qwen3.6-27b-ud-q4kxl (16.4 GiB, DENSE)

Re-enabled purely to measure whether deadlock moves the floor your 2026-09-02
note recorded.

| | 4 devices (2026-09-02) | 5 devices (+deadlock) |
|---|---|---|
| split | `22,28,28,22` | `22,18,18,20,22` |
| **generation** | **1.38 tok/s** | **1.72 tok/s** |
| prompt | — | 1.36 tok/s |
| load | — | 429 s |

GPU shares identical (22% each) in both, so only the CPU redistribution changed.

**+25% — better than I predicted, but the verdict is unchanged.** I expected the
floor to hold flat because deadlock adds RAM, not VRAM, and the bottleneck is
~6 GiB of dense weights crossing RPC on *every* token. Spreading that spill three
ways instead of two clearly helps at the margin, but 1.72 tok/s is still ~15x
slower than the MoE models on this same pool and nowhere near usable.

So the original conclusion stands: **this is an architecture limit, not a tuning
miss.** Dense models want VRAM. Nothing deadlock provides can fix that — only a
bigger card would.

Also confirmed from the metadata: `general.architecture = 'qwen35'`, which this
llama.cpp build (Sep 1) *does* support, with a correct 4-element
`rope.dimension_sections`. The earlier warning about qwen35 models failing is
about a DIFFERENT file — the Ollama-converted `qwen35-27b-cryoai`, which has the
structural SSM problem. This upstream Unsloth conversion loads fine. It was
disabled for being slow, not for being broken.


## Full multi-node lineup (5-device pool, deadlock included)

Run 2026-09-10 00:34. Each model: warmup (excluded, carries the load) + 2 timed runs,
fixed prompt, max_tokens=200, temperature=0. tok/s is llama.cpp's own
`predicted_per_second`.

| model | split | gen tok/s | prompt tok/s | load s | vs README |
|---|---|---|---|---|---|
| `gpt-oss-120b-mxfp4` | `8,20,20,44,8` | **6.47** | 7.27 | 1422.7 | |
| `qwen3-coder-next-ud-q2kxl` | `19,20,20,22,19` | **7.45** | 3.03 | 522.3 | |
| `qwen3-30b-a3b-2507-q4km` | `26,16,16,16,26` | **14.57** | 14.72 | 336.8 | |
| `qwen3-coder-30b-a3b-q4km` | `26,16,16,16,26` | **14.52** | 14.35 | 338.2 | |
| `qwen3-30b-a3b-2507-q3km` | `34,10,10,12,34` | **16.35** | 15.87 | 244.0 | |
| `qwen3-coder-30b-a3b-q3km` | `34,10,10,12,34` | **16.26** | 16.08 | 243.7 | |
