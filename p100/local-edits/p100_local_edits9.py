#!/usr/bin/env python3
"""Brick 5 (2026-10-08): live GPU profiler hooks (PROF=N tune key) + exact single-pass top-k in common_sampler.
Applies to /data/llm/src/llama.cpp-v050; new files come from /data/llm/src/p100-local-files/. Idempotent."""
import shutil, sys

SRC = "/data/llm/src/llama.cpp-v050"
FILES = "/data/llm/src/p100-local-files"


def edit(rel, pairs):
    path = f"{SRC}/{rel}"
    s = open(path).read()
    for old, new in pairs:
        if new in s:
            continue
        n = s.count(old)
        if n != 1:
            sys.exit(f"{rel}: anchor found {n} times: {old[:80]!r}")
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("edited", rel)


for f in ("p100-prof.cu", "p100-prof.cuh"):
    shutil.copy(f"{FILES}/{f}", f"{SRC}/ggml/src/ggml-cuda/{f}")
    print("copied", f)

edit("ggml/src/ggml-cuda/ggml-cuda.cu", [
    ('#include "ggml-cuda/mmvh-pascal.cuh"\n',
     '#include "ggml-cuda/mmvh-pascal.cuh"\n#include "ggml-cuda/p100-prof.cuh"\n'),
    ('    else if (strcmp(key, "MMVH_MC")   == 0) { ggml_cuda_tune.mmvh_mc   = atoi(val); }\n',
     '    else if (strcmp(key, "MMVH_MC")   == 0) { ggml_cuda_tune.mmvh_mc   = atoi(val); }\n'
     '    else if (strcmp(key, "PROF")      == 0) { ggml_cuda_prof_arm(atoi(val)); }\n'),
    ('    ggml_cuda_tune_refresh();\n    ggml_cuda_set_device(cuda_ctx->device);\n',
     '    ggml_cuda_tune_refresh();\n    ggml_cuda_set_device(cuda_ctx->device);\n'
     '    if (ggml_cuda_prof_gen) {\n        ggml_cuda_prof_segment_begin(cuda_ctx->device, cuda_ctx->stream());\n    }\n'),
    ('    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);\n\n'
     '    return GGML_STATUS_SUCCESS;\n',
     '    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);\n\n'
     '    if (ggml_cuda_prof_live) {\n        ggml_cuda_prof_segment_end(cuda_ctx->device, cuda_ctx->stream());\n    }\n\n'
     '    return GGML_STATUS_SUCCESS;\n'),
    ('                ggml_cuda_tune_node_seq[cuda_ctx->device]++;\n',
     '                ggml_cuda_tune_node_seq[cuda_ctx->device]++;\n'
     '                if (ggml_cuda_prof_live && !use_cuda_graph) {\n'
     '                    ggml_cuda_prof_node(cuda_ctx->device, cuda_ctx->stream(), node);\n'
     '                }\n'),
])

edit("ggml/src/ggml-cuda/allreduce.cu", [
    ('bool ggml_cuda_ar_allreduce(\n        ggml_cuda_ar_pipeline * p,\n',
     'static bool ggml_cuda_ar_allreduce_impl(\n        ggml_cuda_ar_pipeline * p,\n'),
    ('    return ok;\n}\n\n#else // defined(GGML_USE_MUSA)',
     '    return ok;\n}\n\n'
     '// PROF=N timing (p100-prof.cuh): mark each device\'s compute stream around the whole AllReduce, both paths.\n'
     'bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline * p, ggml_backend_t * backends, ggml_tensor ** tensors) {\n'
     '    if (!ggml_cuda_prof_live) {\n'
     '        return ggml_cuda_ar_allreduce_impl(p, backends, tensors);\n'
     '    }\n'
     '    for (int i = 0; i < p->n_devices; ++i) {\n'
     '        ggml_cuda_set_device(p->devices[i]);\n'
     '        ggml_cuda_prof_mark(p->devices[i], static_cast<ggml_backend_cuda_context *>(backends[i]->context)->stream(), "~allreduce");\n'
     '    }\n'
     '    const bool ok = ggml_cuda_ar_allreduce_impl(p, backends, tensors);\n'
     '    for (int i = 0; i < p->n_devices; ++i) {\n'
     '        ggml_cuda_set_device(p->devices[i]);\n'
     '        ggml_cuda_prof_mark(p->devices[i], static_cast<ggml_backend_cuda_context *>(backends[i]->context)->stream(), "~after_ar");\n'
     '    }\n'
     '    return ok;\n'
     '}\n\n#else // defined(GGML_USE_MUSA)'),
])
s = open(f"{SRC}/ggml/src/ggml-cuda/allreduce.cu").read()
if '#include "p100-prof.cuh"' not in s:
    i = s.index('#include "allreduce.cuh"')
    s = s[:i] + '#include "p100-prof.cuh"\n' + s[i:]
    open(f"{SRC}/ggml/src/ggml-cuda/allreduce.cu", "w").write(s)
    print("edited allreduce.cu include")

edit("common/sampling.cpp", [
    ('    void set_logits(struct llama_context * ctx, int idx) {\n',
     '    // topk > 0: the chain is known to keep only the top-k candidates (common_sampler_prefilter_k), so pick exactly\n'
     '    // those in one pass over the logits instead of filling all n_vocab entries and partially sorting them\n'
     '    void set_logits(struct llama_context * ctx, int idx, int topk = 0) {\n'),
    ('            GGML_ASSERT(logits != nullptr);\n            cur.resize(n_vocab);\n',
     '            GGML_ASSERT(logits != nullptr);\n'
     '            if (topk > 0 && topk < n_vocab) {\n'
     '                cur.resize(topk);\n'
     '                for (int i = 0; i < topk; i++) {\n'
     '                    cur[i] = llama_token_data{i, logits[i], 0.0f};\n'
     '                }\n'
     '                std::sort(cur.begin(), cur.end(), [](const llama_token_data & a, const llama_token_data & b) {\n'
     '                    return a.logit > b.logit;\n'
     '                });\n'
     '                float floor = cur[topk - 1].logit;\n'
     '                for (llama_token token_id = topk; token_id < n_vocab; token_id++) {\n'
     '                    const float l = logits[token_id];\n'
     '                    if (l > floor) {\n'
     '                        int j = topk - 1;\n'
     '                        while (j > 0 && cur[j - 1].logit < l) {\n'
     '                            cur[j] = cur[j - 1];\n'
     '                            j--;\n'
     '                        }\n'
     '                        cur[j] = llama_token_data{token_id, l, 0.0f};\n'
     '                        floor  = cur[topk - 1].logit;\n'
     '                    }\n'
     '                }\n'
     '                cur_p = { cur.data(), cur.size(), -1, true };\n'
     '                return;\n'
     '            }\n'
     '            cur.resize(n_vocab);\n'),
    ('llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {\n',
     '// 2026-10-08 P100 work: when nothing ahead of top-k in the chain can change a logit (no logit bias or suppressed\n'
     '// tokens, neutral penalties / DRY / top-n-sigma, no mirostat), the chain only ever sees the top-k candidates.\n'
     '// Returns that k, or 0 for the full path.\n'
     'static int common_sampler_prefilter_k(const common_sampler * gsmpl, const llama_vocab * vocab) {\n'
     '    const auto & p = gsmpl->params;\n'
     '    if (p.mirostat != 0 || p.top_k <= 0 || !p.logit_bias.empty()) {\n'
     '        return 0;\n'
     '    }\n'
     '    int32_t n_suppress = 0;\n'
     '    llama_vocab_get_suppress_tokens(vocab, &n_suppress);\n'
     '    if (n_suppress > 0) {\n'
     '        return 0;\n'
     '    }\n'
     '    for (const auto t : p.samplers) {\n'
     '        switch (t) {\n'
     '            case COMMON_SAMPLER_TYPE_TOP_K:\n'
     '                return p.top_k;\n'
     '            case COMMON_SAMPLER_TYPE_PENALTIES:\n'
     '                if (!(p.penalty_last_n == 0 || (p.penalty_repeat == 1.0f && p.penalty_freq == 0.0f && p.penalty_present == 0.0f))) {\n'
     '                    return 0;\n'
     '                }\n'
     '                break;\n'
     '            case COMMON_SAMPLER_TYPE_DRY:\n'
     '                if (!(p.dry_multiplier == 0.0f || p.dry_base < 1.0f || p.dry_penalty_last_n == 0)) {\n'
     '                    return 0;\n'
     '                }\n'
     '                break;\n'
     '            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:\n'
     '                if (p.top_n_sigma > 0.0f) {\n'
     '                    return 0;\n'
     '                }\n'
     '                break;\n'
     '            default:\n'
     '                return 0;\n'
     '        }\n'
     '    }\n'
     '    return 0;\n'
     '}\n\n'
     'llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {\n'),
    ('    auto & cur_p = gsmpl->cur_p; // initialized by set_logits\n\n    gsmpl->set_logits(ctx, idx);\n',
     '    auto & cur_p = gsmpl->cur_p; // initialized by set_logits\n\n'
     '    {\n'
     '        // full path while the reasoning budget is forcing (it rewrites every logit) or the grammar goes first\n'
     '        const bool rb_forcing = rbudget && common_reasoning_budget_get_state(rbudget) == REASONING_BUDGET_FORCING;\n'
     '        const int  topk = rb_forcing || (grammar_first && grammar_should_apply(gsmpl)) ? 0 :\n'
     '            common_sampler_prefilter_k(gsmpl, llama_model_get_vocab(llama_get_model(ctx)));\n'
     '        gsmpl->set_logits(ctx, idx, topk);\n'
     '    }\n'),
])
print("done")
