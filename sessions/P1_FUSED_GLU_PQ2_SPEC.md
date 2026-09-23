# P1 IMPLEMENTATION SPEC — Fused GLU Matvec for PQ2_0 (v9-generation body)
Plan ref: B2_PERF_RECOVERY_PLAN_v2.0.md P1. Author: campaign agent. Date 2026-09-23.

## Opportunity (sized from Phase 0/1 data)
- 434 PQ2_0 matvec dispatches/tok; FFN = 3 matvecs/layer x 64 layers (gate, up, down).
- The graph walk ALREADY has a fused gate+up+GLU hook (ggml-sycl.cpp:6092 ->
  ggml_sycl_mul_mat_glu_mmvq_fused) but it declines for PQ2_0 twice:
  (a) ggml_sycl_mul_mat_vec_q_glu_reorder (mmvq.cpp:4028) hard-returns unless Q4_K.
  (b) ggml_sycl_supports_reorder_mmvq (ggml-sycl.cpp:4038) omits PQ2_0, so the SoA
      reorder never installs and extra->optimized_feature.reorder stays false.
- Fusing gate+up+SWIGLU per layer: removes 64 launches/tok, removes the GLU elementwise
  launch AND one activation round-trip (write gate+up intermediates, read them back),
  and increases per-launch parallelism (Phase 1b verdict = THE bottleneck class).

## PQ2_0 block facts (ggml-common.h:200-207)
- QK_PQ2_0=128, block = fp16 d + uint8 qs[32] (2 bits/el, LE packing), values {0,1,2,3}-> {-1,0,1,2}.
- block_to_q8_1_ratio = 128/32 = 4 (QK8_1=32). vdr_mmvq: whole-block vec_dot, 4 q8_1 chunks -> VDR 4.

## Implementation steps (5 seams, all in existing patterns)
1. quants.hpp: template <> struct block_q_t<GGML_TYPE_PQ2_0> in ggml_sycl_reordered ns:
   traits { qk=128, qi=QI_PQ2_0(=8, 32 bytes/4B-words), qr=QR_PQ2_0(=4), vdr_mmvq=4 }.
   get_block_offset(i, n) = { i * 32, 0 }            (qs region: 32 B/block, all rows)
   get_d_offset(nrows, ncols, i): nblocks = nrows*(ncols/128);
     total_qs = nblocks*32; d at total_qs + i*sizeof(ggml_half).
   block_to_q8_1_ratio() = 4.
   (Follow Q2_K pattern lines 61-82; single-region qs + trailing per-block fp16 scales.)
2. ggml-sycl.cpp: reorder_qw_pq2_0(data_device, size, offset, stream) — one-time SoA
   shuffle kernel: pass 1 gathers qs bytes into region A, pass 2 (or same kernel, second
   range) gathers fp16 d into region B. Copy reorder_qw_q2_k as the pattern (closest
   shape: qs + scale arrays), adapt byte counts (32 B qs + 2 B d per 128 el).
   Wire case GGML_TYPE_PQ2_0 into reorder_qw() (ggml-sycl.cpp:4643 switch).
3. ggml-sycl.cpp:4038: add `case GGML_TYPE_PQ2_0: return true;` in
   ggml_sycl_supports_reorder_mmvq.
4. vecdotq.hpp: reorder vec_dot for PQ2_0 — operator()(vx, bx_offset, d_offset, q8_ptr,
   q8_ds, iqs): read uint32 word from qs region at bx_offset(+lane words), unpack 2-bit
   SWAR (SAME math as vec_dot_pq2_0_q8_1_swar — bit-reversal fixed version), scale by
   fp16 d loaded from d_offset region; accumulate into the 4-chunk dot exactly as the
   Q2_K reordered vecdot does. REUSE the proven SWAR body; only the addressing changes.
5. mmvq.cpp:4028 (ggml_sycl_mul_mat_vec_q_glu_reorder): accept PQ2_0 (keep Q4_K branch),
   `using vec_dot = reorder_vec_dot_q_sycl<GGML_TYPE_PQ2_0>;`, and for ncols_dst==2 use
   the rows_per_sg=2 branch when nrows >= 6272 (FFN gate/up are 17408 — yes).

## Correctness gates (all must pass before any e2e claim)
- G1. Standalone harness (pq2_kernel_diff.cpp pattern): reordered-SoA dot == v9 SWAR dot
  bit-exact on random data + REAL model dumps (pq2_lb*.bin fixtures).
- G2. In-model: GGML_SYCL_PQ2_NO_MMVQ=1 off; run 32-token gen; output must stay coherent
  (same tokens as v9 path) — the 'Paris' anchor.
- G3. A/B with pinned prefill route (MMVQ_DECODE_ONLY=1 both sides), same day/build.
- G4. VTune occupancy of the fused kernel vs 3-kernel baseline (expect > 62%).

## Expected win (honest estimate)
- Removes ~64 matvec launches + ~64 GLU launches + 2x intermediate traffic per layer.
- If in-model FFN BW goes 95 -> ~120 GB/s from co-residency relief alone (review: fused
  quantize was P1c's top co-residency suspect), e2e 17.3 -> ~20-22 t/s. Graph capture
  (P2) stacks on top. Do NOT promise 48 from P1 alone.

## Risks
- The fused kernel is Q4_K-tuned (rows_per_sg=2 at ncols_dst==2); PQ2_0 blocks are 32 B
  vs Q4_K 18 B — SLM sizing constants in launch_mul_mat_vec_q_reorder_glu_impl may need
  a PQ2_0-aware constant. Check static_asserts at compile time, not runtime.
- opt_for_reorder installs the SoA layout lazily on FIRST use; a mis-reorder is silent
  wrong math -> G1 with real dumps is non-negotiable before G2/G3.
- v10 lesson: the SoA quantize path (quantize_and_reorder_q8_1_soa) is shared with Q4_K
  and already bit-validated there — do NOT touch it; only the weight-side reorder is new.
