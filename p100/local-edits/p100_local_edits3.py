#!/usr/bin/env python3
"""Third round of local P100 changes (on top of p100_local_edits.py and p100_local_edits2.py): runtime tuning knobs.

The Pascal switches (AllReduce, its BF16 wire, CUDA graphs, mmvq presets) become one struct that is set from the
environment and, when GGML_CUDA_TUNE_FILE names a file, re-read from it whenever it changes (checked at most every
200 ms, at graph compute). One llama-server can then be measured under every setting without reloading the model.
File format: KEY=VALUE lines, keys AR, AR_BF16, GRAPHS, MMVQ (MMVQ=S,L). Production keeps using the env variables.

Usage: p100_local_edits3.py <llama.cpp source dir>. Each edit must match exactly once.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "ggml/src/ggml-cuda"


def edit(name, old, new):
    p = root / name
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        sys.exit(f"{name}: expected 1 match, found {n}:\n{old[:300]}")
    p.write_text(s.replace(old, new))
    print(f"ok  {name}")


# knob struct in its own header, included only where used (keeps the rebuild to three files)
(root / "p100-tune.cuh").write_text("""#pragma once

// Pascal (P100) switches, see ggml_cuda_tune_refresh() in ggml-cuda.cu. ar: internal AllReduce on pre-Volta;
// ar_bf16: BF16 wire threshold in bytes (-1 = GGML_CUDA_AR_BF16_THRESHOLD/default); graphs: CUDA graphs on pre-Volta;
// mmvq_s/mmvq_l: one-token mat-vec presets for K <= 6144 / longer K.
struct ggml_cuda_tune_knobs {
    int ar      = 0;
    int ar_bf16 = -1;
    int graphs  = 0;
    int mmvq_s  = 0;
    int mmvq_l  = 0;
};
extern ggml_cuda_tune_knobs ggml_cuda_tune;
// Whether GGML_CUDA_TUNE_FILE is set: AllReduce is then initialized on Pascal so the file can switch it at runtime.
bool ggml_cuda_tune_file_active();
void ggml_cuda_tune_refresh();
""")
print("ok  p100-tune.cuh")

edit("ggml-cuda.cu", """#include "ggml-cuda/allreduce.cuh"
""", """#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/p100-tune.cuh"
""")
edit("mmvq.cu", """#include "mmvq.cuh"
""", """#include "mmvq.cuh"
#include "p100-tune.cuh"
""")
edit("allreduce.cu", """#include "allreduce.cuh"
""", """#include "allreduce.cuh"
#include "p100-tune.cuh"
""")

edit("ggml-cuda.cu",
"""static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
""",
"""#include <sys/stat.h>
#include <chrono>

ggml_cuda_tune_knobs ggml_cuda_tune;

static void ggml_cuda_tune_apply(const char * key, const char * val) {
    if      (strcmp(key, "AR")      == 0) { ggml_cuda_tune.ar      = atoi(val); }
    else if (strcmp(key, "AR_BF16") == 0) { ggml_cuda_tune.ar_bf16 = atoi(val); }
    else if (strcmp(key, "GRAPHS")  == 0) { ggml_cuda_tune.graphs  = atoi(val); }
    else if (strcmp(key, "MMVQ")    == 0) {
        int s = 0, l = 0;
        if (sscanf(val, "%d,%d", &s, &l) < 2) {
            l = s;
        }
        ggml_cuda_tune.mmvq_s = s;
        ggml_cuda_tune.mmvq_l = l;
    }
}

bool ggml_cuda_tune_file_active() {
    return getenv("GGML_CUDA_TUNE_FILE") != nullptr;
}

void ggml_cuda_tune_refresh() {
    static bool   env_done = false;
    static double last_check = 0.0;
    static long long last_mtime = -1;
    if (!env_done) {
        env_done = true;
        static const std::pair<const char *, const char *> env_keys[] = {
            {"GGML_CUDA_AR_PASCAL", "AR"}, {"GGML_CUDA_AR_BF16_THRESHOLD", "AR_BF16"},
            {"GGML_CUDA_GRAPHS_PASCAL", "GRAPHS"}, {"GGML_CUDA_MMVQ_PASCAL_CFG", "MMVQ"},
        };
        for (const auto & kv : env_keys) {
            if (const char * v = getenv(kv.first)) {
                ggml_cuda_tune_apply(kv.second, v);
            }
        }
    }
    const char * path = getenv("GGML_CUDA_TUNE_FILE");
    if (path == nullptr) {
        return;
    }
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now - last_check < 0.2) {
        return;
    }
    last_check = now;
    struct stat st;
    if (stat(path, &st) != 0 || (long long) st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec == last_mtime) {
        return;
    }
    last_mtime = (long long) st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        return;
    }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char * eq = strchr(line, '=');
        if (eq == nullptr) {
            continue;
        }
        *eq = 0;
        char * nl = strpbrk(eq + 1, "\\r\\n");
        if (nl) {
            *nl = 0;
        }
        ggml_cuda_tune_apply(line, eq + 1);
    }
    fclose(f);
    GGML_LOG_INFO("%s: AR=%d AR_BF16=%d GRAPHS=%d MMVQ=%d,%d\\n", __func__, ggml_cuda_tune.ar, ggml_cuda_tune.ar_bf16,
                  ggml_cuda_tune.graphs, ggml_cuda_tune.mmvq_s, ggml_cuda_tune.mmvq_l);
}

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_tune_refresh();
    ggml_cuda_set_device(cuda_ctx->device);
""")

# graphs: knob, re-evaluated every call so the file can turn them on and off
edit("ggml-cuda.cu",
"""    // Upstream turns graphs off before Volta; GGML_CUDA_GRAPHS_PASCAL=1 keeps them on to measure that on the P100s.
    static const bool graphs_pascal = getenv("GGML_CUDA_GRAPHS_PASCAL") != nullptr && atoi(getenv("GGML_CUDA_GRAPHS_PASCAL"));
    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA && !graphs_pascal) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }
""",
"""    // Upstream turns graphs off for good before Volta. On the P100s the GRAPHS knob decides (GGML_CUDA_GRAPHS_PASCAL=1
    // or the tune file), re-checked every call so it can be switched at runtime.
    if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
        graph->disable_due_to_gpu_arch = !ggml_cuda_tune.graphs;
    }
""")

# AllReduce: init on Pascal if env asks or a tune file may switch it; per-call knob decides
edit("allreduce.cu",
"""        if (cc < GGML_CUDA_CC_VOLTA && !(ar_pascal && cc >= GGML_CUDA_CC_PASCAL)) {
""",
"""        if (cc < GGML_CUDA_CC_VOLTA && !((ar_pascal || ggml_cuda_tune_file_active()) && cc >= GGML_CUDA_CC_PASCAL)) {
""")

edit("allreduce.cu",
"""    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;
""",
"""    const size_t bf16_threshold = ggml_cuda_tune.ar_bf16 >= 0 ? (size_t) ggml_cuda_tune.ar_bf16 : p->bf16_threshold;
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        bf16_threshold > 0 &&
        input_nbytes >= bf16_threshold;
""")

edit("ggml-cuda.cu",
"""static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
""",
"""static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    // On Pascal the AR knob can send a call back to the meta backend's host-staged fallback.
    if (!ggml_cuda_tune.ar && ggml_cuda_info().devices[comm_ctx->dev_ids[0]].cc < GGML_CUDA_CC_VOLTA) {
        return false;
    }
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
""")

# mmvq presets from the knobs
edit("mmvq.cu",
"""                static const std::pair<int, int> env_cfg = [] {
                    int s = 0, l = 0;
                    const char * e = getenv("GGML_CUDA_MMVQ_PASCAL_CFG");
                    if (e != nullptr && sscanf(e, "%d,%d", &s, &l) < 2) {
                        l = s;
                    }
                    return std::make_pair(s, l);
                }();
                const int pcfg = table_id == MMVQ_PARAMETERS_PASCAL && !has_ids ?
                    (ncols_x <= 6144 ? env_cfg.first : env_cfg.second) : 0;
""",
"""                const int pcfg = table_id == MMVQ_PARAMETERS_PASCAL && !has_ids ?
                    (ncols_x <= 6144 ? ggml_cuda_tune.mmvq_s : ggml_cuda_tune.mmvq_l) : 0;
""")
