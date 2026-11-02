# Qwen3.8-27B on the P100s: what limits tok/s (live analysis 2026-10-08)

Measured on the live Code-mode sessions (features #4 and #5). Tools: router log timings, opencode's session DB,
nvidia-smi sampling, `perf` + `gdb` on node04 (installed for this), and since 02:40 the built-in GPU profiler
(`PROF=N` in `/var/tmp/ggml-tune.txt` on node04 -> `/var/tmp/ggml-prof-<pid>.txt`; no restart needed).

## State at pause (2026-10-08 ~11:30)

LIVE from the next router start (router idle, nothing half-applied): `build-v050-p100d`, `MMVH_MC=104`,
`MMVH_MC_Q6=4`, `spec-draft-n-max = 2`, `reasoning-budget = 16384`, ctx-checkpoints 8, cache-ram 4096, 13G job memory.
vs this morning: thinking ~24 -> ~30.7 tok/s, code ~35 -> ~40-43, 90k context ~20 -> ~22+.

Draft length (job 4501, production kernels): 2 drafts think 30.7 / code 39.8; 3: 28.8 / 40.9; 4: 23.6 / 34.5
(per-position acceptance 69% / 49% / 37% / 28%; a 5-token verify falls outside the 2-4 column kernels).

Still running / queued (results only, never touch production):
- job 4502 `brick9b.sh` (FA tile-kernel parameter sweep): CANCELLED at 11:20, link failed (libcudart/libcublas not
  found): the script exports LD_LIBRARY_PATH=$BUILD/bin:$CUDA_HOME/lib64 BEFORE building, which drops the toolchain lib
  dir llm-env.sh puts there. Fix: move that export below the builds, then resubmit (scratch build build-v050-p100e,
  scratch source /var/tmp/fa-sweep-src on node04).
- job 4503 `brick10.sh`: FP16 vs FP32 cuBLAS for prompt reading, KL divergence + llama-bench. Output brick10-4503.out.
  Decision needed after: switch GGML_CUDA_CUBLAS_COMPUTE_TYPE to f16 (up to ~1.7x prompt speed) if KLD stays tiny.
- While 4503 runs (~20 min from 11:20) a Qwen request waits for node04's GPUs; to free them: `ssh slerm-head scancel 4503`.

Rollback: router-rpc.sh backups `.bak-brick8-20261008` (p100c/MC=4), `.bak-brick6-20261008`, `.bak-brick5-20261008`;
models.ini `.bak-draft2-20261008` (draft 3). Restore + `POST 127.0.0.1:8091/restart` on slerm-head if a router is up.

Resume plan: read 9b/10 results -> deploy the best FA config (rebuild a new dir, never p100d) and the FP16 prompt
decision; then host gaps (mask build, split-state cache), small-op fusion; slot carry across restarts needs the
MTP drafter's state.

## Where the time goes

| What | Measured | Notes |
|---|---|---|
| Thinking volume | 94% of Qwen's output is hidden thinking (Ornith 64-82%) | single steps think 15-40k chars = 3-9 min |
| Template default | `reasoning_effort` = **xhigh** when the client sends none | user: keep it (not a lever) |
| Speculative step | ~100-120 ms per step, 2.5-3.6 tokens per step | draft acceptance ~45-55% on thinking prose, ~80% on code |
| GPUs during generation | full boost 1328 MHz, no throttling, ~175 W of 250 W, 67 C | not a cooling/power problem |
| Memory bandwidth used | **24%** | decode should be bandwidth-bound; it isn't |
| Server main thread | 99% busy: 48% waiting on AllReduce events, 13% on stream sync, 39% host work | one thread drives both GPUs |
| GPU-idle gaps | **~15% of wall time**, one ~14 ms gap per speculative step | host work between passes while both GPUs wait |
| Prompt reading | ~230 tok/s at 6-17k ctx (after the memory fix), ~100 tok/s chunks at 105k+ | |

Inside the 14 ms gaps: CPU sampling over the 248k vocab 35%, meta-backend split bookkeeping 9%
(`ggml_backend_meta_get_split_state`, its cache is wiped every pass), attention-mask build on CPU 8%
(`set_input_kq_mask_impl`, grows with context), KV `seq_rm` of rejected drafts 3%.

### GPU profile, prompt reading at 6-17k (02:40, 48 s, both GPUs identical)

| Share | What |
|---|---|
| ~70% | weight matmuls at n=1024 (cuBLAS after dequantize): ~7.7 TFLOPS = **83% of the P100's FP32 peak** |
| 13.3% | AllReduce: 4.4 ms per 10 MB (bf16) sync through host memory = ~2.3 GB/s |
| 6.3% | flash attention (grows linearly with depth) |
| 2.8% | gated delta net (recurrent layers) |
| 1.8% | gaps between passes |

## Done

- 02:38 **router job 4495 on build-v050-p100b** (brick 5): exact single-pass top-k in the sampler (old vs new:
  27/27 completions token-identical, incl. presence penalty and grammar resampling), live GPU profiler.
  models.ini qwen3.8: `ctx-checkpoints = 8`, `cache-ram = 4096`, `slot-save-path`. router-rpc.sh exports
  `GGML_CUDA_TUNE_FILE`. Backups `*.bak-brick5-20261008`; rollback = LLM_BUILD back to build-v050-p100 + restart.
- 02:35 **4th OOM kill: context checkpoints.** The server keeps up to 32 per slot (one per request), each with this
  hybrid model's recurrent state (~150+ MB): anon memory grew 8.7 -> 11.9 GB over the chat. Capped at 8 (outputs
  unchanged: opencode prompts only grow, so only the newest checkpoints are ever used).
- 02:25 **memory.high throttling.** Slurm sets `memory.high` = `--mem` (10 GiB) as well as `memory.max`; the job sat
  at 10.8 GB, so the kernel throttled every allocation and evicted code pages (read back from NFS): 3.3M throttle
  events, 2.95M major faults, one step read 886 tokens in 549 s. Side effect: the router's `/models` lock backed up
  (96 stuck connections), so the orchestrator showed every model `cold`. Fixed live; new jobs get 13G for both.
- 02:05 third OOM kill: Slurm also caps the nested `step_batch/user` cgroup; raised.
- Slurm memory cap 10G -> 13G (`#SBATCH --mem=13G`, backup `.bak-mem10g-20261008`).
- **Code-mode titles/tags** use the model the router already has loaded (code-mode `393604f`, deployed to Open
  WebUI; backup of the old function on cryoai `~/code_mode.deployed-before-393604f.py`). Before: every new Code chat
  swapped Qwen out for an Ornith title request (~5 min, cache lost).
- `watch-session SESSION_ID` on VM 129: read-only live view of thinking + tool calls (watch-ornith's TUI opens blank).

- 08:55 (user OK) **thinking cap** for qwen3.8: `reasoning-budget = 16384` + a short "time to act" message in models.ini
  (effort stays xhigh). Why: at 03:31 one feature-#5 step thought ~26k tokens (105k chars, 19.5 min) without acting,
  filled the 131k context, was truncated, and the 2h10m run ended with no code. Applies at the next router start.

## Brick 6 (in test, job 4497 on node04)

- **v4 multi-column verify kernel** (MMVH_MC=2/4): R rows per warp share each run's activations. 50/50 correct vs CPU;
  on a busy GPU 1.4-1.6x faster than stock MMVQ at 4 tokens with R=4. The big lever: verify (4 tokens) costs ~2.3x a
  1-token pass now, physically it should be ~1.2-1.3x (same weights read once). At that cost: est. ~40-45 tok/s on
  thinking, ~50-55 on code (from ~25 / ~30). Next: activations staged in shared memory so R can grow, + IQ4_XS.
- **FA tile GQA padding** (FA_GQA_PAD=4/8): Qwen3.8's 6:1 GQA packs 2 heads, so every K/V head is read 3x per pass;
  padding to 8 reads it once. Only for <= 8 tokens (prompt tiles already share K/V). ~+5% at 30k, ~+12% at 90k (est).

## Brick 6 result (LIVE from the next router start: build-v050-p100c, MMVH_MC=4)

| Workload | stock | v4, 4 rows | gain |
|---|---|---|---|
| thinking, short ctx | 24.2 | 28.9 | +19% |
| code, short ctx | 34.4 | 42.4 | +23% |
| thinking, 90k ctx | 20.2 | 21.9 | +9% |

Idle-GPU 4-token matmuls: Q4_K 209 -> 129 us, Q5_K 219 -> 144, Q6_K 241 -> 156. FA padding helps only 1-token batches
(4-token verify attention is arithmetic-bound in the tile kernel), so it stays off. 90k profile after: v4 verify 35%,
attention 21%, IQ4_XS verify (stock) 10%, small ops 7%, AllReduce 6%, idle 9%.

## Brick 7 result (not deployed)

Occupancy variants (fewer rows, or a register cap for 3-4 blocks/SM) were all slower than 4 rows uncapped: rows
sharing activations is the only lever that paid. IQ4_XS through v4 (separate lookups per run) gained nothing on the
kernel and lost end to end (code 42.1 -> 38.4), so it stays off. Brick 8 = v5: activations in shared memory, 6-8 rows.

## Still to do (needs a restart, a test window, or your call)

1. **cuBLAS FP16 compute for prompt matmuls** (`GGML_CUDA_CUBLAS_COMPUTE_TYPE`, set to f32 earlier "most
   faithful"). Matmuls are ~70% of prompt time and sit at 83% of the FP32 peak; FP16 is 2x. Up to ~1.7x prompt
   speed. Precision trade-off: needs a KLD test, then your decision.
2. **GPU-side sampling under tensor split** (disabled upstream for SPLIT_MODE_TENSOR). Remaining CPU sampling cost
   after brick 5 is the logits copy + the chain on 20 candidates.
3. **Attention mask on the GPU / incremental** (~1 ms/step at 30k, more when deep).
4. **Exact split-state cache** in the meta backend (validate entries by their sources instead of wiping). ~1%.
5. **Drafting tuned for thinking text** (2 drafts or p-min when acceptance is low). Needs an A/B.
6. **Verify pass kernel (brick 4)**: 4-token verify runs on MMVQ; decode-phase profile will size it.
7. **Prompt reading at depth**: flash attention grows linearly; profile again past 60k.
8. **Two slots (`--parallel 2 --kv-unified`)** so a title/tags request for another chat uses the idle slot instead
   of evicting a long chat's cache (now that titles go to the loaded model). Needs a test with MTP + tensor split.
9. **Seamless restarts**: slot save/restore works for the main model, but the MTP drafter's state is not in the
   file, so drafting would restart cold after a restore. Test before relying on it.
10. **Hardware**: the two P100s sit on different CPU sockets (no peer-to-peer) and one is in an x8 slot; both on
    one socket's x16 slots could allow direct GPU-to-GPU copies for the 128 AllReduces per token.
11. opencode TUI via watch-ornith opens blank (stalls after its update check). Not investigated; watch-session
    covers watching.
