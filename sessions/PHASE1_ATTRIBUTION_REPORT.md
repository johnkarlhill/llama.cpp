# PHASE 1 ATTRIBUTION REPORT — B2 PQ2_0 on B70 (Sep 22 2026)
Build bc9772a44 (v3-clean + per-node wall timer, GGML_SYCL_NODE_TIMER). 100-token decode,
temp 0 seed 42, per-node queue-sync timing. CSV: /tmp/node_timer.csv (6.0MB, 104 graph dumps).

## Measured per-token breakdown (sync-tax inflated total 92.8 ms/tok vs 74.6 ms/tok clean e2e)
category                              n/tok   ms/tok     %
MMVQ FFN (up+gate+down, 64 layers)    199.7   47.78    51.5%
MMVQ attn/ssm projections             216.3   22.60    24.3%
FWHT-1024 hadamard folds              268.3    5.08     5.5%
MUL_MAT other (GDN z/gates etc)        99.8    2.97     3.2%
output head (248320x5120 PQ2)            1     2.93    3.2%
elementwise ops (MUL/ADD/GET_ROWS/SCALE/UNARY/GLU/CPY/CONCAT/SET_ROWS/CONT)
                                      ~978    ~8.4     ~9.1%
FLASH_ATTN_EXT                         16.6    1.10     1.2%
NORMs (RMS/L2)                         99.8    0.79     0.9%
ROPE                                   33.3    0.40     0.4%

## THE HEADLINE: MMVQ execution IS the bottleneck in-model.
- MMVQ total: 73.3 ms/tok measured (79%). Non-MMVQ: ~19.5 ms upper-bound (~10 ms real).
- Effective in-model BW: FFN matvecs = 95 GB/s; attn+head = 104 GB/s.
- BUT v5 isolated bench (same kernel, FFN 17408x5120, b1) = 175 GB/s.
- => In-model the kernel delivers only ~54% of its isolated rate.
- Campaign's earlier derivation ('v5-rate predicts 24.5 t/s; gap must be non-matmul')
  was WRONG: the gap lives INSIDE in-model MMVQ execution.

## Hypothesis scoreboard (FINAL for this cycle)
- H1 non-matmul cost: DEMOTED — real non-MMVQ ~10ms/tok, not 43.
- H4 FWHT: CLEARED — 5.1 ms/tok (5.5%), 18.9 us/call; not the villain.
- H2 launch overhead: CLEARED — elementwise+norms+rope+glue = ~12 ms incl tax.
- NEW H8: in-model MMVQ efficiency collapse (95 vs 175 GB/s). Suspects:
  (a) L2 cache thrash — 4.5GB FFN weights stream evicts everything; isolated bench had
      hot L2; (b) concurrent GDN/attn kernels interleaving break MMVQ saturation;
  (c) clock/power throttle under sustained mixed load; (d) q8_1 quantize kernels
  between matvecs displacing weights from cache.

## Path to 48 t/s
48 t/s = 150 GB/s effective across all matmuls. Currently 95-104.
Fixing H8(a/d) — cache-resident scheduling, fused quantize, larger tiles — is the lever.
MMVQ kernel micro-opt (v5 is already best-correct) is NOT the lever; the same kernel
demonstrates 175 GB/s when conditions are right. The problem is the ENVIRONMENT the
kernel runs in, not the kernel.


## PHASE 1b — VTune gpu-hotspots verdict (Sep 22, run vt_pq2, PQ2 tg48)
- GPU busy only 32.5% of elapsed (14.8s wall, 4.8s GPU) under profiling.
- WHILE BUSY: XVE Array Stalled/Idle = 91.2%. The GPU is starved, not throttled.
- mul_mat_vec_pq2_0_q8_1_sycl: SIMD width 16, occupancy 49.4/62.6/71.3%, SIMD util 100%.
- Q6_K control reproduces 22.78 t/s same day/build/env -> no global slowdown; PQ2-specific.
- CONCLUSION: in-model 95-104 GB/s (vs 175 isolated) is an OCCUPANCY/parallelism limit.
  The v5 kernel at FFN shapes leaves ~half the 256-XVE array unfilled and cannot hide
  DRAM latency; isolation benchmarking pipelines submits back-to-back (queue depth),
  in-model dependency chains serialize kernels -> latency exposure every launch.
- H8 REFINED: not L2 thrash, not throttle. It is per-launch parallelism deficiency +
  no inter-kernel overlap in decode dependency chains.
- LEVER: raise per-launch parallelism (more row-groups per kernel, multi-tensor batched
  launches, fuse q8_1 quantize into matvec, split-K) OR enable inter-kernel overlap
  (multiple queues / graph-level reordering). Kernel micro-opt unchanged: v5 math is fine.


## Phase 1c — occupancy A/B results (Sep 22, late)
| variant | serialized harness GB/s | VTune occ | tg128 |
|---|---|---|---|
| v5/lut 1 warp/grp (A) | 156.3 | 72.5-76.9% (harness) | 13.32 |
| v7/lut 8 warps/grp (B) | 156.8 | (same harness env) | 13.29 |
| v8 ILP-2 (C) | 117.3 (REGRESSION) | - | not benched |
Harness VTune: occ 72.5-76.9%, XVE stalled 81-82%.
In-model VTune: occ 49.4%, XVE stalled 91.2% (SAME kernel+build).
Conclusions:
- Launch shape (1 vs 8 warps/group) does NOT change occupancy or speed.
- ILP-2 hurts (register pressure). v8 abandoned.
- Isolated kernel achieves 72-77% occ + 156 GB/s serialized. The SAME kernel in-model
  runs at 49.4% occ + 95-105 GB/s. The gap is environmental: co-resident kernels
  (q8_1 quantize, elementwise, GDN) share XVEs and pollute L2 during decode.
- The 156-175 GB/s 'isolated' number is NOT reachable in-model by kernel changes alone.
  The in-model fix must reduce co-residency damage: fuse quantize into matvec, batch
  tensors per launch, or reorder graph to give matvecs exclusive GPU slices.
- Hardware-limit check: 13.3 t/s at 95 GB/s effective. If co-residency were eliminated
  AND matvec-only time were 73ms -> 156 GB/s -> ~24 t/s matmul-limited -> plus ~10ms
  other => ~21 t/s e2e. 48 t/s needs kernel + co-residency fixed, still 2x away.
Next: (a) fused quantize+matvec prototype (biggest single lever), (b) VTune with
platform-overview to measure concurrent-kernel overlap directly.


## Phase 1d — L2-cold A/B (Sep 22, kd_cold.cpp via ggml_debug_pq2_0_run which=3)
- hot  (1 tensor reused):     141.2 GB/s
- cold (3 tensors rotating):  111.9 GB/s   (-21%)
- ICL=0 submission serialization: NO effect (PQ2 13.27, Q6_K 22.77).
- Complete degradation chain: 175 (hot+pipelined) > 156 (hot serialized) > 141 (hot,
  fresh run) > 112 (L2-cold weights) > ~100 (in-model = cold + quantize churn).
- Q6_K对比: 20.88GB/token at ~478 GB/s effective = DRAM-saturating WHILE COLD. Same
  1-warp/row launch. Q6_K's template inner loop hides cold-DRAM latency; v5's single-
  block-per-lane-per-iter does not (insufficient memory-level parallelism).
- FINAL ROOT CAUSE: v5 lacks memory-level parallelism to hide DRAM latency on cold L2.
  L2 reuse in microbenchmarks masked it; Q6_K never had the problem.
- v9 SPEC (next session): template-style inner loop for PQ2 - lane strides 32 blocks but
  processes >=2 blocks/iter with explicit prefetch (load i+1 while dp4a i), SWAR decode
  (LUT raised register pressure in v8). Target: cold-L2 BW >= 150 GB/s.
