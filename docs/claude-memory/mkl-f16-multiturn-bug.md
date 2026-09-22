---
name: mkl-f16-multiturn-bug
description: "F16 cache + MKL FA multi-turn corruption — FIXED in commit 13f503255"
metadata:
  node_type: memory
  type: project
  originSessionId: 0064f37a-aff4-46be-add6-ac4c5d6cf942
---

# MKL F16 Multi-Turn Corruption Bug — FIXED

## Root Cause
The MKL kernel was writing F16 K/V directly into the ggml tensor without copying to a dense buffer first. During multi-turn, the accumulated cache had different stride patterns that the MKL GEMM kernel didn't handle correctly.

## Fix (commit 13f503255)
Always copy F16 K/V to a dense buffer before passing to MKL GEMM — same as the dequant path does for quantized types. This normalizes strides and prevents the corruption.

## Fixed Gate (same commit)
Added `mask && !sinks && Q->ne[0] >= 64` to prevent MKL from firing on unsupported cases.

## Status
Bug is resolved. PR #25025 includes the fix. See [[project_tps_sycl_perf]] for current PR state.
