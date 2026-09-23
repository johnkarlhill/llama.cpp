# B2 PERF RECOVERY PLAN v2.0 — Post-Review Roadmap
Date: 2026-09-23. Supersedes v1.1. Incorporates kimi-k3 adversarial review (deleg_4e2e9439, 163s, 8 API calls).

## Targets (corrected arithmetic)
- Roofline: 6.7 GiB / 608 GB/s = **84.5 t/s @100% DRAM**.
- 48 t/s = 48 × 7.19 GB = **345 GB/s effective = 57% of peak** (NOT 150 GB/s — v1.x conflated
  the v9 cold-L2 kernel gate with the e2e requirement; downstream math was off 2.3×).
- Feasibility: Q6_K control sustains 79% of peak in-model cold on this exact box ⇒ 57% is
  physically achievable but demands Q6_K-class memory behavior from a 2.13 bpw format.
- **Interim gate: 28-32 t/s** (v9 in-model ~200 GB/s + non-matmul halved). Stretch: 48.
- Caveat: non-weight traffic (KV + activations) never itemized; real requirement may be
  390-430 GB/s. Measure before locking the target.

## Baseline discipline (applies to EVERY A/B from now on)
- Pin prefill route (GEMM vs MMVQ) — the 17.3 t/s baseline is a *configuration*, not just a
  kernel. Unpinned comparisons are meaningless.
- Clean e2e anchor: 17.3 t/s = 57.8 ms/tok. Per-node-timer numbers are upper bounds
  (timer syncs per node = serializes the pipeline it measures). MMVQ share "~60-73 ms", not 79%.

## Phases
### P1. Multi-tensor batched matvec (TOP LEVER — in progress)
434 matvec launches/tok each expose cold-DRAM latency at launch. Batch per-layer up/gate/down
(and attn projections) into single launches: amortizes latency exposure, raises per-launch
parallelism (Phase 1b verdict: per-launch parallelism deficiency).
- Sub-lever: 2-4 rows/warp for narrow attention tensors (v6's idea, done correctly — no
  reqd_sub_group_size(32); hardware SG = 16).
### P2. Launch-count reduction / graph capture
4000+ launches/tok is STRUCTURALLY incompatible with 48 t/s (5.2 µs/launch budget; L0 overhead
3-8 µs). Level-0 graph capture or command-batching of the decode subgraph. Launch overhead was
"cleared" under a sync-inflated timer — treat as UNMEASURED, re-measure properly.
### P3. Elementwise fusion
978 elementwise launches = 8.4 ms/tok. Fuse norm+scale+GLU chains into matvec epilogues/
prologues and GDN producers. Simultaneously cuts launches (P2) and removes co-resident kernels
blamed for the 49.4% vs 72-77% occupancy collapse (P1c).
### P4. FWHT-1024 batching
301 launches, 5.1 ms/tok, 18.9 µs/call. Group shape-compatible folds. Bounded gain 2-3 ms/tok.
### P5. Non-weight traffic census (measurement, run early/parallel)
Itemize KV + activation DRAM bytes/tok. At 2.13 bpw activations are a much larger byte
fraction than for Q6_K. Decides whether 48 needs 345 or 390-430 GB/s.

## Demoted / dead
- Fused q8_1 quantize into matvec (v10): DEMOTED. Quantize math never bit-validated ("MINI
  returns 0" = localized, not root-caused); re-quantizes the row once per weight-row (wrong
  shape). If revived: quantize-once fused into the *producer* (elementwise/GDN), not consumer.
- v10 stays parked. LUT, v6, v8, MTP, NVFP4: dead (prior findings stand).

## Red-team counter-experiments (run when P1 lands or stalls)
- R1. Co-residency hypothesis never directly tested: run decode with everything except FFN
  matvecs stubbed (keep graph shape), VTune the matvec occupancy. No recovery ⇒ batched-launch
  depth, not co-residency, is the real lever.
- R2. L0 Windows submission artifact: A/B `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`,
  (stretch) Linux run. The "dependency-chain serialization" may be driver behavior.
- R3. v9 8-blocks/lane variant: one build cycle; settles whether the remaining 2.8× to
  Q6_K-parity lives in the kernel (18B stride coalescing suspect, never retired) or environment.
- R4. q8_1-noise proof (cheap): v9 reading F32 src1 directly (v10 launch shape minus quantize);
  output flipping back to "Paris" proves the noise claim.
- R5. Doc hygiene: reconcile Q6_K effective BW citation (478 vs 504 GB/s) — pick one byte
  accounting.

## Success criteria
- P1 lands: in-model MMVQ ≥ 200 GB/s ⇒ ≥ 24 t/s e2e.
- Interim (P1+P2+P3): 28-32 t/s.
- Full stack + R-experiments resolving favorably: 48 t/s.
