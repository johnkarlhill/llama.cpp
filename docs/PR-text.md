# SYCL: HYBRID Flash Attention — Dequantize Quantized KV for oneDNN SDPA

## Overview

This PR adds a **HYBRID** flash attention path that dequantizes quantized KV caches (Q4_0, Q5_0, Q5_1, Q8_0, BF16, F32) to F16, then routes them through oneDNN's fused SDPA primitive (MatMul→Scale→Mask→SoftMax→MatMul as a single hardware XMX kernel).

The key insight: our dequant-to-F16 pipeline produces exactly the dense F16 buffers that oneDNN's fused SDPA primitive needs. Wire them together, and quantized models get hardware fused attention — without requiring the SDPA path to natively support quantized types.

## Dependencies

This PR bridges two existing works:

- **PR [#25025](https://github.com/ggml-org/llama.cpp/pull/25025)** (MKL FA): Provides the dequant-to-F16 pipeline and MKL GEMM fallback. Must merge first.
- **PR [#25222](https://github.com/ggml-org/llama.cpp/pull/25222)** (oneDNN SDPA, @hmscider): Provides the native F16 oneDNN SDPA path. The `build_sdpa()` graph-construction code in this PR is the same attention graph used by #25222, reproduced here so HYBRID can operate independently. When #25222 merges, its native F16 ONEDNN path slots in above HYBRID in the dispatch order.

When both base PRs are merged, this PR activates automatically — no code conflicts.

## Build

Requires oneDNN (DNNL):

```
cmake --preset x64-windows-sycl-release -DGGML_SYCL_DNN=ON -DGGML_SYCL_F16=ON
cmake --build build-x64-windows-sycl-release --config Release -j 16
```

**`-DGGML_SYCL_F16=ON` is critical.** Without it, MKL GEMM silently falls back to fp32, which is 2-3× slower.

## Dispatch Architecture

HYBRID sits between MKL (GEMM-based, all types) and the eventual ONEDNN path (F16-only, from #25222):

| Priority | Path | KV Types | Description |
|---|---|---|---|
| **1. ONEDNN** | (reserved for #25222) | F16 only | Native oneDNN SDPA — single fused XMX kernel. |
| **2. HYBRID** | `fattn-hybrid.cpp` | Q4_0, Q5_0, Q5_1, Q8_0, BF16, F32 | Dequant K/V → oneDNN SDPA → permute to ggml layout. **New in this PR.** |
| **3. MKL** | `fattn-mkl.cpp` | All types | MKL GEMM flash attention (from #25025). Handles ALiBi, logit softcap, no-mask cases that SDPA rejects. |
| **4. TILE/VEC** | existing | All types | Fallback for decode and small prefill. |

## HYBRID Pipeline

Each FA call for a quantized KV layer:

1. **Dequant K** to dense F16 `[d, seq, Hkv]` (using persistent buffer)
2. **Dequant V** to dense F16 `[d, seq, Hkv]` (or alias K if shared buffer)
3. **Copy Q** to dense F16 `[d, q, H]` (pool alloc, Q is always F32 in ggml)
4. **Build/run oneDNN SDPA**: MatMul→Divide→Add(mask)→SoftMax→MatMul, f16 output, single fused XMX kernel
5. **Permute** to ggml dst layout `[d, H, q]`
6. **On exception**: fall back to MKL GEMM

## Gate Conditions

HYBRID fires when ALL conditions are met:

- oneDNN compiled in (`GGML_SYCL_DNNL=1`)
- `GGML_SYCL_FA_ONEDNN=1` (default; set to 0 to disable all oneDNN-based paths)
- KV is NOT both native F16 (those go to ONEDNN when #25222 is present)
- F16 mask, no attention sinks
- head_dim ≥ 32, n_kv ≥ 1024, Q batch ≥ 32
- max_bias == 0, logit_softcap == 0
- V->ne[0] == K->ne[0], Q->ne[3] == 1, GQA divides evenly

Control via environment variables:
- `GGML_SYCL_FA_ONEDNN=1` (default) — enables HYBRID
- `GGML_SYCL_FA_ONEDNN=0` — disables HYBRID, falls back to MKL
- `GGML_SYCL_ENABLE_MKL_FA=0` — disables MKL too, pure TILE/VEC baseline

## Performance (Intel Arc Pro B70 / Battlemage)

Qwen 3.6 27B Q5_K_XL, 32K prefill, 155K context:

| Mode | KV Cache | Path | Prefill t/s | vs Baseline |
|---|---|---|---|---|
| Stock (TILE) | F16 | TILE | 335 | 1.00× |
| Stock (TILE) | Q8_0 | TILE | 332 | 1.00× |
| MKL-only | Q8_0 | MKL GEMM | 633 | 1.91× |
| **This PR** | Q8_0 | **HYBRID** (dequant→SDPA) | **824** | **2.48×** |

Gemma 4 31B Q4_K_XL, 147K context (interleaved head layout):

| Mode | KV Cache | Path | Prefill t/s | vs Baseline |
|---|---|---|---|---|
| Stock (TILE) | Q8_0 | TILE | 178 | 1.00× |
| **This PR** | Q8_0 | **HYBRID** | **440** | **2.48×** |

Key findings:
- HYBRID is **30% faster** than MKL GEMM alone (824 vs 633 t/s) — the fused SDPA kernel is more efficient than 3 separate GEMM + softmax launches
- HYBRID delivers the same throughput as native F16 ONEDNN SDPA — the dequant step is not a bottleneck
- Gemma interleaved head layout sees the same **2.48×** speedup
- Token generation throughput is maintained at baseline levels

## Test Coverage

Validated across 6 models × 3 KV quant types × 3+ multi-turn conversations:

| Model | Architecture | KV types tested | Status |
|---|---|---|---|
| Qwen 3.6 27B Q5_K_XL | Dense, D=256 | F16, Q8_0, Q4_0, mixed | ✓ |
| Qwen 3.6 35B MoE Q4_K_XL | MoE A3B, D=256 | Q8_0 | ✓ |
| Gemma 4 31B Q4_K_XL | Dense, interleaved | F16, Q8_0, mixed | ✓ |
| Gemma 4 26B MoE Q4_K_XL | MoE A4B, interleaved | Q8_0 | ✓ |
| Qwen 3.5 9B Q6_K_XL | Dense, D=256 | F16 | ✓ |
| Gemma 4 12B Q4_K_XL | Dense, interleaved | Q8_0 | ✓ |

All combinations produce coherent output across multi-turn conversations with normal draft acceptance rates (60-73%). Zero TILE fallbacks on any prefill-shaped call.

`test-backend-ops test -o FLASH_ATTN_EXT`: **0 FAIL**.

## Files Changed

| File | Change |
|---|---|
| `fattn-hybrid.hpp` | **New** — HYBRID gate + SDPA partition declarations |
| `fattn-hybrid.cpp` | **New** — dequant K/V → SDPA pipeline + `build_sdpa()` |
| `fattn.cpp` | Modified — HYBRID dispatch, gate conditions, env var controls |
| `ggml-sycl.cpp` | Modified — `GGML_SYCL_FA_ONEDNN` env var |
| `common.hpp` | Modified — extern declaration |

## Bugs Fixed

1. **Permute formula**: SDPA output is contiguous `[d, q, H]`, ggml dst is `[d, H, q]`. Prior code had `h` and `t` swapped in the offset calculation, scrambling data for all positions except `(head=0, token=0)`.

2. **Pool alloc race**: `ggml_sycl_pool_alloc` frees GPU memory in its destructor without synchronizing the SYCL stream. Added `stream->wait()` before function return in `fattn-hybrid.cpp` to prevent pool memory from being freed and reused while the GPU is still executing SDPA + permute kernels.

3. **MKL gate missing guards**: The MKL kernel was firing on cases it can't handle (no mask, attention sinks, head_dim < 40). Added `mask && !sinks && Q->ne[0] >= 64` to all XMX gate paths.

## Credits

The `build_sdpa()` graph-construction function uses the same oneDNN Graph API pattern established by **@hmscider** in PR [#25222](https://github.com/ggml-org/llama.cpp/pull/25222). The `sdp_primitive_kernel_t` is what makes this level of performance possible — replacing 3 kernel launches with a single hardware fused XMX kernel.

The dequant-to-F16 pipeline is built on the MKL FA infrastructure from PR [#25025](https://github.com/ggml-org/llama.cpp/pull/25025).

## Author's Note

This PR represents a collaboration between human expertise and AI-assisted development. All architectural decisions — the dequant-to-SDPA bridge concept, the gate structure, and the environment variable controls — were designed and directed by @johnkarlhill. The AI tooling (Claude Code) served as a force multiplier for implementation, debugging, and testing, operating under explicit human guidance and code review at every step.

The six-model, three-quant-type, multi-turn validation campaign was designed and executed manually on real Intel Arc Pro B70 hardware. Every performance number reported here comes from that testing, not synthetic benchmarks.

Co-Authored-By: Claude Code with DeepSeek-v4-Pro
