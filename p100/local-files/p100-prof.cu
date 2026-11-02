// Live GPU timeline for the P100 work (2026-10-08), see p100-prof.cuh.

#include "p100-prof.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

int ggml_cuda_prof_gen  = 0;
int ggml_cuda_prof_live = 0;

namespace {

struct prof_rec {
    int         ev;
    std::string key;
};

struct prof_dev {
    int  gen       = 0;
    bool active    = false;
    long remaining = 0;
    long segments  = 0;
    double total_ms = 0.0;
    std::vector<cudaEvent_t> pool;
    std::vector<int>         free_ev;
    std::deque<prof_rec>     chain;                      // stream order; front = oldest not yet read back
    std::map<std::string, std::pair<double, long>> acc;  // key -> (ms, intervals)
};

int      arm_n = 0;
prof_dev devs[GGML_CUDA_MAX_DEVICES];

void record(prof_dev & d, cudaStream_t stream, std::string key) {
    int i;
    if (d.free_ev.empty()) {
        cudaEvent_t e;
        CUDA_CHECK(cudaEventCreate(&e));
        d.pool.push_back(e);
        i = (int) d.pool.size() - 1;
    } else {
        i = d.free_ev.back();
        d.free_ev.pop_back();
    }
    CUDA_CHECK(cudaEventRecord(d.pool[i], stream));
    d.chain.push_back({i, std::move(key)});
}

// Attribute every interval whose closing event has completed to the key of its opening event.
void drain(prof_dev & d, bool wait) {
    while (d.chain.size() >= 2) {
        const prof_rec & a = d.chain[0];
        const prof_rec & b = d.chain[1];
        if (wait) {
            CUDA_CHECK(cudaEventSynchronize(d.pool[b.ev]));
        } else {
            const cudaError_t st = cudaEventQuery(d.pool[b.ev]);
            if (st == cudaErrorNotReady) {
                (void) cudaGetLastError();
                break;
            }
            CUDA_CHECK(st);
        }
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, d.pool[a.ev], d.pool[b.ev]));
        auto & e = d.acc[a.key == "~tail" && ms > 2.0f ? std::string("~tail>2ms") : a.key];
        e.first  += ms;
        e.second += 1;
        d.total_ms += ms;
        d.free_ev.push_back(a.ev);
        d.chain.pop_front();
    }
}

void release_chain(prof_dev & d) {
    for (const auto & r : d.chain) {
        d.free_ev.push_back(r.ev);
    }
    d.chain.clear();
}

void write_summary(int device, const prof_dev & d) {
    char path[128];
    snprintf(path, sizeof(path), "/var/tmp/ggml-prof-%d.txt", (int) getpid());
    FILE * f = fopen(path, "a");
    if (f == nullptr) {
        return;
    }
    char ts[32];
    const time_t now = time(nullptr);
    strftime(ts, sizeof(ts), "%F %T", localtime(&now));
    fprintf(f, "== %s device %d: %ld segments, %.1f ms of GPU timeline\n", ts, device, d.segments, d.total_ms);
    std::vector<std::pair<std::string, std::pair<double, long>>> v(d.acc.begin(), d.acc.end());
    std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second.first > b.second.first; });
    for (const auto & kv : v) {
        fprintf(f, "%10.2f ms %5.1f%% %9ld x %9.1f us  %s\n", kv.second.first,
                100.0 * kv.second.first / std::max(1e-9, d.total_ms), kv.second.second,
                1000.0 * kv.second.first / std::max(1L, kv.second.second), kv.first.c_str());
    }
    fclose(f);
}

// "blk.12.ffn_up.weight" -> "ffn_up.weight", "Qcur-12" -> "Qcur-#"
std::string simplify(const char * s) {
    if (strncmp(s, "blk.", 4) == 0) {
        const char * q = s + 4;
        while (*q >= '0' && *q <= '9') {
            q++;
        }
        if (*q == '.') {
            s = q + 1;
        }
    }
    std::string out;
    bool digits = false;
    for (; *s; ++s) {
        const bool d = *s >= '0' && *s <= '9';
        if (!d) {
            out += *s;
        } else if (!digits) {
            out += '#';
        }
        digits = d;
    }
    return out;
}

std::string node_key(const ggml_tensor * node) {
    std::string k = ggml_op_desc(node);
    k += ' ';
    if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[0] && node->src[1]) {
        k += simplify(node->src[0]->name);
        k += ' ';
        k += ggml_type_name(node->src[0]->type);
        k += " n=" + std::to_string(node->src[1]->ne[1]);
    } else if (node->op == GGML_OP_GET_ROWS && node->src[0] && node->src[1]) {
        // 2026-10-17: which gather is slow (Flash-Next GET_ROWS averaged 157 us). ne1 bucketed (KV views grow).
        int64_t b = 1;
        while (b < node->src[0]->ne[1]) {
            b *= 2;
        }
        k += simplify(node->name);
        k += ' ';
        k += ggml_type_name(node->src[0]->type);
        k += " ne0=" + std::to_string(node->src[0]->ne[0]) + " ne1<=" + std::to_string(b) +
             " ne2=" + std::to_string(node->src[0]->ne[2]) + " rows=" + std::to_string(ggml_nelements(node->src[1]));
    } else {
        k += simplify(node->name);
    }
    return k;
}

} // namespace

void ggml_cuda_prof_arm(int n_segments) {
    arm_n = n_segments;
    ggml_cuda_prof_gen++;
}

void ggml_cuda_prof_segment_begin(int device, cudaStream_t stream) {
    prof_dev & d = devs[device];
    if (d.gen != ggml_cuda_prof_gen) {
        d.gen = ggml_cuda_prof_gen;
        if (d.active) {
            ggml_cuda_prof_live--;
        }
        release_chain(d);
        d.acc.clear();
        d.total_ms  = 0.0;
        d.segments  = 0;
        d.remaining = arm_n;
        d.active    = arm_n > 0;
        if (d.active) {
            ggml_cuda_prof_live++;
        }
    }
    if (!d.active) {
        return;
    }
    drain(d, false);
    record(d, stream, "~seg_begin");
}

void ggml_cuda_prof_segment_end(int device, cudaStream_t stream) {
    prof_dev & d = devs[device];
    if (!d.active) {
        return;
    }
    record(d, stream, "~tail");
    d.segments++;
    if (--d.remaining <= 0) {
        drain(d, true); // the one blocking wait, at the end of the run
        write_summary(device, d);
        release_chain(d);
        d.active = false;
        ggml_cuda_prof_live--;
    }
}

void ggml_cuda_prof_node(int device, cudaStream_t stream, const ggml_tensor * node) {
    prof_dev & d = devs[device];
    if (d.active) {
        record(d, stream, node_key(node));
    }
}

void ggml_cuda_prof_mark(int device, cudaStream_t stream, const char * key) {
    prof_dev & d = devs[device];
    if (d.active) {
        record(d, stream, key);
    }
}
