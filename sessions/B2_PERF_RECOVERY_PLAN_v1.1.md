# PQ2/Bonsai-2 Performance Recovery Plan — v1.1 (execution revision)

**Revision:** 1.1 — supersedes the v1.0 external analysis (`GPT.md`). v1.0's central thesis is
**accepted unchanged**; this revision corrects four hypotheses against the measured session
record and the fork source, folds two already-answered experiments into confirmations, and
re-sequences execution to extract free attribution from existing instrumentation first.

**Objective (unchanged):** 48+ t/s decode on Bonsai-2 PQ2_0 on Intel Arc Pro B70, or a measured
proof that the target is unattainable under current architecture.

---

## 0. What v1.0 got right (kept verbatim in intent)

- **STOP OPTIMIZING DECODE UNTIL ATTRIBUTION EXISTS.** The project is attribution-limited, not
  implementation-limited. All seven correct decode variants (template/SWAR/LUT/v3/v4/v5/v7)
  converge to 13.3-13.7 t/s e2e while v5 alone sustains 175 GB/s isolated.
- The No-Go list: no more LUT/SWAR/byte_perm/decode-correctness work. Agreed — all exhausted.
- Success = 48+ t/s OR a measured proof of unattainability. Agreed.

## 1. Baseline facts (unchanged, all measured)

| Fact | Value | Source |
|---|---|---|
| Q6_K same box | 22.87 t/s, ~504 GB/s, 83% BW | Sep 20 matrix |
| PQ2_0 e2e | 13.3-13.7 t/s, ~91-105 GB/s, 16-17% BW | Sep 20-21 |
| PQ2_0 roofline | ~84 t/s @ 2.13 BPW / 608 GB/s | Sep 19 |
| v5 isolated | 175 GB/s (0.136 ms, FFN 17408x5120 b1) | Sep 21 isolation |
| template isolated | 109 GB/s | Sep 21 isolation |
| v6 isolated | 281.6 GB/s = **half-work artifact**, genuinely broken | Sep 21, fixed harness |
| Dequant+GEMM e2e | 320 ms/tok vs MMVQ 72 ms/tok | Sep 21 bisect |
| Derived gap | v5-rate matmuls predict 24.5 t/s; e2e 13.4 => **~10 ms/token non-matmul** | Sep 21 |

---

## 2. CORRECTIONS TO THE v1.0 HYPOTHESIS TREE

### C1. H3 (ncols=2 path dominance) — DEMOTED, likely wrong

v1.0 claimed generation "repeatedly traverses switch_ncols(..., 2)". Fork source
(`mmvq.cpp:3077`): the multi-col dispatch fires only when `src1_ncols > 1 && <= 8`; at batch-1
decode `src1_ncols == 1`, so every dense matmul takes the **single-column path** all kernels
target. The full session record contains zero log evidence of ncols=2 dominating gen — the only
`switch_ncols` mentions are from the Sep 19 wiring review. The ncols_dst=2 tuning that exists
(Q4_K reorder rows_per_sg=2) targets multi-token shapes, not our decode.

**Action:** killed as a hypothesis unless Phase 1's trace shows otherwise (one grep of
GGML_SYCL_DEBUG output settles it — see Phase 0). Phase 5 of v1.0 is withdrawn as a standalone
phase.

### C2. H2's "MoE-like behavior" — ALREADY DISMISSED (Sep 20)

Bonsai-2 is a **dense** Qwen3.5-hybrid (GDN attention + hadamard folds, no expert tensors).
The `launch_mul_mat_vec_q_moe` PQ2_0 dispatch (mmvq.cpp:3507) is generic table coverage, never
reached — verified Sep 20 22:55: "dense, so the MoE expert-matmul theory is out." Launch-overhead
concern itself SURVIVES (GDN + 401 folded weights add graph nodes), but the MoE framing is cut.

### C3. Phase 6 (MMVQ vs dequant+GEMM) — partially answered, demoted to confirmation

Measured Sep 21 during the correctness bisect: `GGML_SYCL_PQ2_NO_MMVQ=1` (dequant+oneDNN GEMM)
= **320 ms/tok vs 72 ms/tok** MMVQ. The XMX-prefers-big-GEMM hypothesis starts 4.4x behind.
Re-run once with the fixed kernels as a one-line confirmation; do not budget investigation time.

### C4. Fact-#2 framing: the attribution datapoint already exists

The doc presents "~90-105 GB/s effective" as an open question. Sharpen it: **v5 isolated = 175
GB/s; e2e effective = ~91-105 GB/s.** The matmul execution itself is not the gap — roughly 65%
of token wall time (72 ms/tok of which matmul-rate accounts for ~29 ms) accrues OUTSIDE the
PQ2 matmuls. Phase 1 confirms and itemizes this; it does not discover it.

---

## 3. ADDITION: PHASE 0 — FREE ATTRIBUTION FROM EXISTING INSTRUMENTATION (Day 1)

The fork already carries debug switches that produce a rough op picture with **zero new code**:

- `GGML_SYCL_DEBUG=1` — per-dispatch logs incl. the PQ2_0 `switch_ncols` call sites (7 files
  carry the macro; mmvq dispatch logs every multi-col branch attempt)
- `GGML_SYCL_PQ2_DUMP=1`, `GGML_SYCL_PQ2_LANEDBG=1`, `GGML_SYCL_PQ2_NO_MMVQ=1`,
  `GGML_SYCL_PQ2_V6=` — kernel-path bisect switches
- Upstream `--dump-dot` graph export for node/op census (call counts per token, no timing)

**Phase 0 checklist (all cheap, all same-day):**
1. `GGML_SYCL_DEBUG=1` decode run on B2 -> grep dispatch log: confirm `ncols_dst=1` everywhere
   (settles C1/H3 in minutes) and count PQ2 matvec dispatches/token.
2. Count graph ops/token via graph dump (FWHT-1024 instances, GDN ops, norms, ropes).
3. Server timing deltas already banked: MMVQ 72 ms/tok, NO_MMVQ 320 ms/tok, Q6_K ~44 ms/tok.

**Known gap:** the SYCL backend has **no per-node timing today** (no chrono/perf hooks in
ggml-sycl.cpp compute loop). Existing switches give counts, not milliseconds. The only genuinely
new instrumentation required for attribution is a per-node timer — that is Phase 1, and it is
smaller than v1.0 estimated because op-census comes free from Phase 0.

---

## 4. REVISED HYPOTHESIS TREE (post-corrections)

| ID | Hypothesis | Prior (v1.0) | Revised | Basis |
|---|---|---|---|---|
| H1 | Large non-MMVQ token cost | High | **High (primary)** | 10 ms/tok derived; 43 ms/tok unexplained vs matmul rate |
| H2 | Launch/scheduling overhead (GDN+FWHT graph complexity) | High | **High** | MoE part dismissed; graph-node count real |
| H4 | FWHT cost | Medium-High | **High — THE unmeasured Tier-1 suspect** | 110+ FWHT-1024 invocations/token est.; zero ms data |
| H6 | q8_1 activation prep | Medium | Medium | unmeasured |
| H5 | GDN cost | Medium | Medium | unmeasured |
| H3 | ncols=2 dominance | High | **Dismissed pending Phase 0 grep** | source + record contradict it |
| — | Dequant+GEMM alternative | (Phase 6) | **Effectively closed** | 320 vs 72 ms/tok measured |
| H7 | Remaining MMVQ opportunity | Low | Low | v5 = ceiling for design; 3 variants converge |

---

## 5. EXECUTION PLAN (re-sequenced)

### Day 1 — Phase 0 (free attribution)
- GGML_SYCL_DEBUG decode trace on B2; confirm ncols_dst=1; dispatch census.
- Graph dump census: ops/token per class (FWHT, GDN, norm, rope, matmul).
- Output: `phase0_census.txt`.

### Day 2-3 — Phase 1 (per-node timer) + Phase 3 (FWHT micro-bench)
- Add per-node timing wrapper in the SYCL compute loop (`ggml_time_us` around node compute,
  aggregated by op + name into a CSV). Deliverable: `bonsai_token_profile.csv`
  (Operation, CallsPerToken, TotalMS, MSPerCall, PercentToken).
- `fwht_bench` — FWHT-128/256/512/**1024** isolated: ms/token, GB/s, invocations/token.
  B2 needs the 1024 case (we restored it in the fork, commit 05658dd0f, after upstream's table
  skipped it). Critical question unchanged: **can FWHT explain 10+ ms/token?**
- Conditions: 100 decode tokens, temp 0, fixed prompt+seed, `SYCL_CACHE_PERSISTENT=0` for A/B,
  oneAPI setvars required on the box (0xC0000135 otherwise).

### Day 4-5 — Phase 2 (token-cost accounting)
- Aggregate: MMVQ / FWHT / attention / GDN / sampling / graph-runtime / copies / other.
- Critical question: **is MMVQ dominant?** (Expected answer from C4: no.) If confirmed ->
  MMVQ work stays frozen per the No-Go list.

### Week 2 — confirmations + counters
1. Re-run NO_MMVQ once with fixed kernels (C3 confirmation only).
2. Launch-count audit from the Phase 1 timer (kernel name, launches/token, avg us).
   If 500+ launches/token -> graph-fusion campaign justified.
3. VTune/GPA counters on the top measured consumer (EU occupancy, XMX util, L2 hit, DRAM BW,
   stall reasons) — hardware counters, not guesses.

### Week 3 — attack the highest measured consumer
- Isolated microbench of that consumer; re-test e2e; iterate.

### Endgames (unchanged from v1.0, minus ncols=2 as a primary)
- A: MMVQ dominates -> v8/persistent/multi-row/fused experts.
- B: FWHT dominates -> shared-memory FWHT, fusion, hadamard caching.
- C: Launch overhead dominates -> graph fusion, kernel batching, persistent decode graph.
- D: GDN/q8_1 dominates -> dedicated path work.

---

## 6. NO-GO LIST (v1.0 list, adopted + extended)

No more: LUT decode variants, SWAR rewrites, byte_perm work, decode correctness churn,
random-block validation loops, MMVQ-only optimization without attribution — all per v1.0.
**Extended:** no ncols=2 specialization work unless the Phase 0 grep shows multi-col dispatch
in decode (expected: it will not); no dequant+GEMM investigation (measured 4.4x worse).

## 7. DEFINITION OF SUCCESS (unchanged)

48+ t/s tg128, or a measured proof of unattainability. Next milestone = the Bonsai-2
token-time accounting report, not another kernel.

---

## Provenance

Corrections C1-C4 derived from: fork source `ggml/src/ggml-sycl/mmvq.cpp` (dispatch conditions
at :3077, MoE table at :3507), session record `20260916_150404_b4c9d10a` (Sep 20 22:55 dense
verification; Sep 21 02:10 NO_MMVQ bisect; Sep 21 14:05 isolation + harness fix), and the
campaign chronology `PQ2_B2_CAMPAIGN_CHRONOLOGY.md`.
