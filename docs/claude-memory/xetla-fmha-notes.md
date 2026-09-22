---
name: xetla-fmha-notes
description: "The Xe flash-attention acceleration landscape — what beats what, and the pitfalls to design around before writing a fused-dequant FMHA kernel"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 0064f37a-aff4-46be-add6-ac4c5d6cf942
  modified: 2026-07-23T22:39:47.958Z
---

# Xe Flash Attention Acceleration Landscape

Captured 2026-07-07 while scoping "is there anything faster than SDPA?" Deferred research is Task #31 (fused-dequant FMHA). This file exists so we walk into that kernel already knowing the traps.

## The performance ordering (Battlemage / Xe2)

- **TILE/VEC (stock XVE)** — shader-unit kernels, no XMX. Prefill decays with context. Baseline.
- **MKL GEMM (#25025)** — dequant→F16→oneMKL GEMM + our own online softmax. XMX via GEMM. Handles ALL K/V types + shapes SDPA can't (no mask, ALiBi, MLA, non-mult-of-64 head_dim via TILE fallback). This is the **portable / Alchemist-safe** path (never touches the fused SDP microkernel).
- **oneDNN SDPA (#25222)** — fused `sdp_primitive_kernel_t`, single XMX kernel, never spills S=QK^T to HBM. **Faster than MKL.** F16 KV only.
- **HYBRID (#25312)** — our dequant→F16 feeding hmscider's SDPA. Gives quantized models SDPA speed. Same BMG-only ceiling as SDPA.

## Is anything faster than SDPA?

- **For F16 KV: no, practically.** SDPA's systolic path IS Intel's hand-tuned DPAS microkernel. ESIMD, XeTLA, and oneDNN brgemm all sit on the same DPAS primitives — they land at ~SDPA. Beating Intel at their own matmul is ~10-20% in narrow cases, huge maintenance, not worth it.
- **For quantized KV (our actual workload): yes — ONE structural win.** HYBRID materializes dequant K/V as dense F16 in HBM, then SDPA reads it back. That F16 write+read of the whole KV cache per prefill is pure bandwidth waste. A **fused-dequant FMHA kernel** dequants each K/V tile in-register and feeds it straight to DPAS — never writes F16 to HBM. oneDNN SDPA **structurally cannot** do this (needs dense F16 inputs). This is how CUDA flash-attn handles quantized KV.

## Honest uplift estimate

Bounded, **~10-25% at long context**. At our throughput (~948 t/s F16 prefill) attention is largely XMX-bound, not dequant-bound, so the F16-materialization saving is an optimization, not a 2×. Weigh against cost: a hand-written Xe FMHA kernel is the single biggest effort on this project.

## Approach candidates

### joint_matrix CONFIRMED dead on BMG (Jul 18, re-confirmed Jul 23)

Full battery of 5 tile configs tested on Arc Pro B70 with oneAPI 2026.0: u8 (8x8x16, 8x16x16, 16x16x16) and fp16 (1x8x16, 8x8x16). ALL five configurations throw EXCEPTION at runtime with hardcoded tile dimensions. The SPIR-V `CooperativeMatrixKHR` extension does not support DPAS on BMG.

Filed as IGCIT issue #1501. Intel engineer previously confirmed `matrix_combinations` query returns msize=0/ksize=0 per spec (must hardcode sizes for m<=8), but hardcoding is not sufficient on BMG.

DG2 has a working `joint_matrix` FA kernel (2340 t/s, 3605/3605). BMG does not.

- **Register-S + cross-lane softmax + high occupancy = 2340 t/s at pp512 (1.05x vs TILE at 2237).** Wins at every prompt length tested: pp1024 2001 vs 1922, pp2048 1520 vs 1448, pp4096 1009 vs 958.
- **3605/3605 test-backend-ops on DG2.** Zero correctness regressions.
- **No SLM for scores.** Each lane owns one full column of the 8x8 QK^T DPAS fragment (all 8 query rows of col=lane). S stays in registers; online-softmax row max/sum via reduce_over_group (m/l/corr also in registers, replicated). The S store→SLM→read round-trip, post-QK barrier, and S/ml SLM buffers are all eliminated.
- **Occupancy was the dominant lever.** Freeing SLM alone (1661→1559 regression) — the reduce_over_group sync point stalls the array. The win only comes after raising sub-group count from 8→16 (SLM budget 48→60KB freed by register-S), so 16 independent DPAS streams hide the reduction latency. Same principle we used for query tiling.
- **Register layout is correct.** DG2 accumulator: each lane = one full column of 8×8 fragment, row-major within lane.

**Implications for us:**
- Task #13 (joint_matrix benchmark on BMG) is no longer academic — someone proved it works on DG2. BMG with 2x XMX could see a bigger gap.
- A joint_matrix path could sit alongside MKL in the funnel: both pure SYCL kernel approaches (no oneDNN dependency), but joint_matrix would target shapes MKL doesn't handle (MHA, non-mult-of-64 head_dim).
- The register-S technique should map directly to XeTLA FMHA (Task #31), avoiding the round-trip that makes HYBRID's materialization wasteful.

### ESIMD xmx::dpas CONFIRMED working on BMG (Jul 18 smoke test)

Ran a standalone ESIMD DPAS test (`smoke_dpas.cpp` in `C:\llama.cpp-build-sycl\xetla`) on Arc Pro B70. Key findings:

- **Compiled and ran without crashing.** No driver exception, unlike `joint_matrix` which threw exceptions on BMG for the standard `8x8x16` tile.
- **Produced results.** The kernel executed `xmx::dpas<8, 1, float>` (fp16 inputs, float accumulator, exec_size=8) successfully. Output was 10.0 instead of expected 16.0 — a systematic VNNI packing error in our test code (all 8 elements identical, zero variance), not random hardware error.
- **ESIMD bypasses the broken IGC JIT.** `joint_matrix` goes through IGC's JIT compiler which can't emit correct DPAS for BMG. ESIMD uses pre-compiled intrinsics that map directly to hardware instructions — same path oneDNN uses, and oneDNN works.

**Implications:**
- **ESIMD is the path.** `xmx::dpas` works on BMG (compiled+ran in smoke test), `joint_matrix` doesn't. ESIMD is lower-level — maps one-to-one with hardware XMX instructions, no CooperativeMatrixKHR SPIR-V layer in between.
- **XeTLA is viable.** Uses ESIMD intrinsics, not `joint_matrix`. Same path that works.
- **ESIMD is the highest-ceiling path for fused-dequant FMHA.** Direct register control, in-register dequant between tiles, zero API overhead. With 2x XMX throughput on BMG vs DG2, a well-tuned ESIMD FA kernel could exceed the DG2 `joint_matrix` result of 2340 t/s at pp512.

1. **XeTLA** (Intel's CUTLASS-equivalent) — ships an `fmha` example, THE fastest hand-tuned Intel attention. Template-heavy, Xe-HPC/Xe2 focused, big integration lift. **Primary candidate.**
2. **ESIMD** (`sycl::ext::intel::esimd`) — direct `xmx::dpas` + manual register/SLM tiling. This is what oneDNN's kernel is underneath. Max control, max work.
3. **oneDNN ukernel/brgemm API** — compose own FA loop over tuned brgemm. Marginal vs the already-fused primitive.

## Cheap wins BEFORE writing any kernel

1. `ONEDNN_VERBOSE=2` — confirm SDPA always dispatches the **systolic** kernel, never silently the `larger_partition` fallback on our shapes. That fallback is the numerically-wrong Alchemist path AND slower; catching a silent fallback on BMG could be a big free swing.
2. **Primitive/graph caching** — don't pay oneDNN graph construction per call.

## Pitfalls to design the kernel around (learned the hard way)

- **Per-arch validation trap.** Xe-HPG (Alchemist) SDPA is numerically broken in oneDNN ≤3.12.2 (fix only in unreleased 3.13 `main`, code-read but UNVERIFIED). A hand-written kernel needs its OWN per-arch NMSE validation on every target — same trap, no free ride.
- **Padded seq-view KV stride.** A real KV cache presents `K->nb[2] = padded seq stride` (view into a 2×-seq buffer), NOT contiguous. Distinguish it from true Gemma interleave (`nb[2] < ne[1]*nb[1]`). This bug bit BOTH MKL and HYBRID dequant. The fused kernel's tile loads must get this right from day one. See [[tps-monitor-project]] bug #2.
- **head_dim coverage.** Must handle the full set {40,64,72,80,96,112,128,256,512}. MKL GEMM couldn't do non-mult-of-64 (72/80/96 crash) → gated to `%64`. A custom kernel must tile these explicitly instead of gating them out.
- **`sizeof(sycl::half)==2`** on this platform; read device fp16 buffers as raw `uint16` + manual convert to dodge host alignment.
- **GQA / gqa_ratio.** MKL needed `gqa_ratio>=2` (MHA quant crashed). A tile kernel handles GQA by K/V head broadcast in the load — design for it, don't gate it.

**Why:** Records the full "what's faster than SDPA and why" analysis so the eventual XeTLA FMHA work (Task #31) starts from known pitfalls, not from scratch. See [[tps-monitor-project]] for PR state and the MKL bug list, [[mkl-fa-workflow]] for build/test recipes.
