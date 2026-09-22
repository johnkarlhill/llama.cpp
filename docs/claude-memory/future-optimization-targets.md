---
name: future-optimization-targets
description: "Post-attention optimization roadmap for Q4_K_M on SYCL/B70, saved for when existing PRs clear"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0064f37a-aff4-46be-add6-ac4c5d6cf942
  modified: 2026-07-19T18:47:06.668Z
---

# Future Optimization Targets (post-PR-clear)

**Status:** Research only. Cannot open more PRs until #25025 and #25874 merge.

## Context

Q4_K_M is the most common quant. It uses 256-element superblocks with hierarchical scales+mins, making it harder to vectorize than Q4_0/Q8_0. SYCL already has the K-quant reorder (separates scales/mins from quants for coalesced loads), which is why Q4_K_M often beats naive Q8_0. But the weight matmuls themselves are still running on vector paths, not XMX.

## Optimization Priority Stack

| Priority | What | How | Effort |
|---|---|---|---|
| 1 (done) | Attention prefill | #25025 (MKL FA) + #25874 (ONEDNN SDPA quants) | Done/near-done |
| 2 (research) | Fused dequant FMHA | XeTLA FMHA kernel | Task #31, research phase |
| 3 (next target) | XMX-accelerated FFN/QKV weight matmuls | oneMKL GEMM or hand-rolled DPAS intrinsics (NOT joint_matrix -- dead on BMG with oneAPI 2026.0, see [[xetla-fmha-notes]]) | High effort, high payoff |
| 4 (diminishing) | Decode MMVQ tuning | Already near bandwidth wall on B70 (~90% util, Qwen 27B ceiling ~21.5 t/s) | Low ROI |

## XMX for Weight Matmuls -- The Problem

The attention PRs make XMX work for attention scores (Q*K^T, S*V) where both operands are freshly dequantized F16. Weight matmuls multiply against quantized weights directly; the dequant has to happen inside the kernel or be fused.

CUDA's MMQ kernels for Q4_K dequant in shared memory, then multiply with activations as F16 using tensor cores. The SYCL equivalent would be:
1. Load Q4_K blocks into SLM
2. Dequant to F16 (or keep as int for DPAS)
3. Feed to DPAS intrinsics or oneMKL GEMM for the actual multiply

`joint_matrix` is confirmed non-viable on BMG with oneAPI 2026.0 (rejects standard Intel DPAS tile 8x8x16). Options are oneMKL GEMM (like #25025's approach) or raw DPAS intrinsics.

## Concrete Starting Point

When PRs clear: profile a full prefill pass with oneAPI tools to measure attention vs. weight matmul time breakdown. If attention drops from dominant to small fraction after #25874, the weight matmuls become the bottleneck worth attacking.

## Source

Grok analysis, 2026-07-18. Key insight: "Getting a competitive XMX-aware (or oneMKL/oneDNN-backed) path for Q4_K_M weight GEMMs would be a major step."
