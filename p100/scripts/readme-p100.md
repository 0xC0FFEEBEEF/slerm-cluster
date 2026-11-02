
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
