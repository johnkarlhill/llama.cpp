# Phase 0 Census — B2 PQ2_0 (Ternary-Bonsai-2-27B-PQ2_0.gguf, 24-token decode trace)
Date: Sep 22 2026. Build: cand-build-b2, GGML_SYCL_DEBUG=1, level_zero:0, temp 0 seed 42.
Trace: /tmp/pq2_b2_trace.log (37.7 MB, 11,426 total lines with dispatch prints)
Gen rate during trace: 12.8 t/s (includes synchronous debug-print tax; clean e2e = 13.4)

## Per-token op census (measured, /24)
- MUL_MAT: 1,444.5/tok total, decomposed as:
  - PQ2_0 matvec dispatches: 434.4/tok (384.4 single-col + 33.3 ncols=2 + 16.7 ncols=4)
  - Hadamard DENSE mul_mat (prism.hadamard.1024 as F32 [1024x1024] src0): 300.9/tok
  - remainder ~709/tok: GDN projections (z/gates), norms-derived, misc F32xQ and F32xF32
- FLASH_ATTN_EXT: 31.3/tok | SSM_CONV: 42.0/tok | GATED_DELTA_NET: 38.0/tok
- ROPE: 62.7/tok | GLU: 125.3/tok | RMS_NORM: 362.1/tok | GET_ROWS: 266.8/tok

## FINDING #1 (CRITICAL): FWHT kernel is NEVER invoked — 0 fwht launches.
  The hadamard fold runs as 301 dense F32 [1024x1024] MUL_MAT calls/token
  (dst shapes [1024,5],[1024,6],[1024,17],[1024,10],[1024,210],[1024,55]).
  The GGML_HINT_SRC0_IS_HADAMARD fast path is NOT taken in this build/route.
  ~0.6 GFLOP/token + 301 extra launches/token of skinny low-occupancy GEMMs.

## FINDING #2: PQ2 multi-col dispatch DOES fire in decode (50/tok, ncols=2 x33 + ncols=4 x17)
  driven by GDN node grouping (src1 ne1=2/4), NOT user batch. So v1.0 H3 was half-right:
  multi-col exists but is 11.5% of PQ2 dispatches; single-col (384/tok) dominates.

## FINDING #3: launch counts/token ~= 1445 MUL + ~715 MUL + 362 RMS + 349 ADD + 276 SCALE
  + 267 GET_ROWS + 231 UNARY + 208 CPY + 125 GLU + 94 CONCAT + ... => well over 4,000
  kernel launches/token. H2 (launch overhead) is very much alive.

## Hypothesis scoreboard after Phase 0
- H4 FWHT: CONFIRMED ACTIVE — but as dense-matmul fallback, not FWHT kernel (worse than feared)
- H2 launch overhead: CONFIRMED plausible (4,000+ launches/tok)
- H3 ncols=2 dominance: REFUTED as 'dominant' (11.5%), CONFIRMED present (GDN-induced)
- H1 non-matmul cost: consistent (matmul count alone can't explain 72 ms/tok at v5 rates)
NEXT (Phase 1): per-node timer to convert counts->ms. FWHT hint path fix = likely quick win.


## CORRECTION (post-write): Finding #1 REVERSED.
Deeper trace analysis: all 7,222 hadamard MUL_MAT dispatches complete 'direct done' with NO
dense op_mul_mat inside => ggml_sycl_op_fwht DID take the fast path (fwht.cpp:276 case 1024).
The FWHT kernel IS running; it simply prints nothing. So: 301 FWHT-1024 launches/token,
NOT dense GEMMs. H4 (FWHT cost) remains a Tier-1 suspect but as kernel-time, not fallback.
The dense-matmul claim in the first draft of this report is withdrawn.
