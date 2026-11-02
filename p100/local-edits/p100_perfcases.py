#!/usr/bin/env python3
"""Adds Qwen3.8-27B one-token decode shapes to test-backend-ops perf mode (P100 kernel work, 2026-10-07)."""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "tests/test-backend-ops.cpp"
s = p.read_text()
old = """        test_cases.emplace_back(new test_l2_norm_batch(GGML_TYPE_F32, { n, 16, 16, 1 }, 4, 1e-12f, true));
    }


    return test_cases;
}"""
new = """        test_cases.emplace_back(new test_l2_norm_batch(GGML_TYPE_F32, { n, 16, 16, 1 }, 4, 1e-12f, true));
    }

    // P100 work (slerm, 2026-10-07): Qwen3.8-27B one-token decode shapes, per GPU under tensor split (m = rows, k = K)
    // and under layer split (17408), plus the halved output head.
    for (ggml_type type_a : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0}) {
        for (auto mk : std::vector<std::pair<int64_t, int64_t>>{{8704, 5120}, {5120, 8704}, {17408, 5120}, {5120, 17408}, {124160, 5120}}) {
            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, mk.first, 1, mk.second, {1, 1}, {1, 1}));
        }
    }

    return test_cases;
}"""
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new))
print("ok  test-backend-ops.cpp")
