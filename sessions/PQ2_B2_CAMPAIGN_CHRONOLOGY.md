# The PQ2_0 / Bonsai-2 Kernel Campaign — Complete Chronological Record

**Session:** `20260916_150404_b4c9d10a` (Hermes, Discord thread #jinx / "Llama.cpp and Doom")
**Window:** 2026-09-16 15:04 CT → 2026-09-21 23:30 CT
**Machine:** John-PC-2024 (192.168.10.142) — MSI B650 + Ryzen 7 7700 + Intel Arc Pro B70 (Battlemage, Xe2), Windows host; agent works from Ubuntu-OpenCode WSL2 via SSH. Dev tree: `C:\llama.cpp-build-sycl\llama.cpp` (Windows) ↔ `~/projects/llama.cpp-sycl` (WSL git fork).
**Model files:** `E:\B70\Ternary-Bonsai-27B-Q2_0_g64.gguf` (gen-1, 6.7 GiB), `E:\B70\Ternary-Bonsai-27B-Q2_g64.gguf` (Bonsai-2 PQ2_0, 2.13 BPW).

---

## 0. THE GOAL (as set by the user, evolving over the campaign)

1. **Sep 16 (origin):** Evaluate whether a "Doom-technique" low-bit decode (integer-native, no FP16 staging) could double decode throughput of a 27B dense model on the B70. Answer established: decode at batch-1 is bandwidth-bound; the current path is already at 83% of the 608 GB/s wall; the *byte lever* (lower-BPW formats) is the real multiplier.
2. **Sep 19 (recon):** After PrismML released ternary Bonsai (Qwen3.8-27B, {-1,0,+1} weights): *"What we need to work on is developing a performant SYCL kernel for B2's default quants"* — PQ2_0 (2.13 BPW) and PTQ1_0 (1.76 BPW).
3. **Performance bar (Sep 20, formalized):** ~48 t/s tg128 ≈ 60% of the 608 GB/s pipe. Explicitly stated: *"13t/s is NOT performant... Goal is not met"*, *"13t/s is unacceptable. That isn't a fix, that's a joke."*
4. **Correctness prerequisite (discovered en route):** make GPU MMVQ produce *correct* output on Bonsai-2 PQ2_0 (it was emitting garbage tokens while the CPU backend was clean).

**Roofline table (established Sep 19, the campaign's north star):**

| Path | BPW | GB/token | Decode ceiling |
|---|---|---|---|
| Q6_K (current prod) | 6.5625 | 17.9 | 33.98 t/s measured |
| MXFP4/IQ4_XS | 4.25 | 11.6 | ~52 t/s |
| Ternary Q2_0_g64 | 2.25 | 7.7 | ~79 t/s |
| PQ2_0 (B2 format) | 2.13 | 7.3 | ~84 t/s |
| PTQ1_0 (1-bit) | 1.76 | 6.0 | ~102 t/s |

---

## PHASE A — RESEARCH: the Doom thesis and the physics (Sep 16)

**A1. Hardware ceilings pinned (15:04–15:30).**
- Web research: B70 = 608 GB/s (256-bit, 19 Gbps GDDR6), 367 TOPS dense INT8 on XMX, 22.94 TFLOPS FP32, XMX FP16 ≈ 183.5 TFLOPS (derived 2:1 INT8:FP16 ratio, cross-checked 4096 ops/clk × 32 cores × 2.8 GHz).
- Key derivation: user's measured 33.98 t/s × ~17.9 GB/token = 608 GB/s = **100% of measured bandwidth** → decode is a memory-stream problem; ALU-side changes cannot double it.

**A2. Doom source research (15:11).** Researched ID's `M_FIXED.C` 16.16 fixed-point math (FRACUNIT, FixedMul shift). Conclusion mapped to GPU: stock SYCL decode **already runs the Doom path** — LUT→INT8→dp4a with exact INT32 accumulation, no FP16 staging (`vecdotq.hpp:976`). The datatype win was already banked; the lever left is bytes/token.

**A3. Correction logged (15:31):** batched *prefill* does NOT dequant in registers — it materializes a full FP16 tensor buffer and hands to oneDNN/MKL (read-4.25 → write-16 → GEMM).

**A4. Verdict delivered (15:39):** "2× decode dream physically blocked; 2× lives in bytes." Options ranked: requant to 4.25 BPW (+40-50%, zero kernel work), or ternary (2.25 BPW → ~79 t/s ceiling, biggest lever on the card).

**A5. Prior-work audit (15:29–15:32).** Session forensics recovered the Aug 31–Sep 1 MXFP4 DPAS kernel work: unit tests 4/46/64/128 PASS with exact MAC counts, corr at P=4096 = 0 bad, max err 0.000000 on the B70. IGC lessons in hand (host-side 64-lane scale replication; dead-store poisoning; OOB guards).

**A6. Handoff (16:33):** User moved implementation planning to OpenCode; a planning document was written to the llama.cpp fork. Sep 16 work ended with the physics established.

---

## PHASE B — RECON & SPEC (Sep 19, 00:24–06:10)

**B1. PrismML release intel (00:24).** Ternary Bonsai-27B (Qwen3.8-based): RTX 5090 = 143 t/s, M5 Max = 46.8 t/s (their peak-gen figures); quality 83.9 vs 85.4 full-precision = 98.2% retention.

**B2. Format availability (00:28, corrected twice).** First read "CUDA-only" was wrong: the **`Q2_0_g64`** variant (2.25 BPW, official llama.cpp format) runs mainline across CPU/Metal/Vulkan/**SYCL**. Only PrismML's preferred `PQ2_0` (2.13 BPW) and `PTQ1_0` (1.76 BPW) packings are fork-only.

**B3. Remote recon of John-PC-2024 (03:51–04:02, user away in Greece; "Go for it!").**
- Both prod servers healthy; E: has 490 GB free; no Bonsai files on disk.
- SYCL fork already contains `GGML_TYPE_TQ2_0`, Q2_0 kernels, `LLM_ARCH_QWEN35`.
- **Downloaded gen-1 `Ternary-Bonsai-27B-Q2_0_g64.gguf` (6.7 GB) and benched it on the dev build:**
  - **pp512 = 1134.63 t/s** (faster than Q6_K's 957 pp4096 — the 2-bit format flies in prefill)
  - **tg128 = 15.28 t/s** — only 19% of the ~79 t/s roofline, *slower than Q6_K despite moving 2.3× fewer bytes*
- **Control same build/harness: Q6_K = 22.82 t/s tg128** → the stock Q2_0 decode path wastes ~80% of available bandwidth.

**B4. INT2-native check (04:29).** Xe2 XMX DPAS supports INT8/INT4/**INT2** multiply with 16/32-bit accumulation (INT2/INT4 at 4× INT8 ops/clk). Ternary {-1,0,+1} is a perfect 2-bit signed encoding. Fallback: unpack 2-bit→INT8→dp4a (already proven Aug 31).

**B5. Fork source audit (06:03, scoped campaign).** PrismML fork cloned to `~/projects/llama.cpp-prism` @ `9a9394a`:
- SYCL backend exists (122 files, incl. `fwht.cpp`) but **zero PQ2_0/PTQ1_0 references**.
- CUDA decode exists (`vecdotq.cuh:809-1000`): `vec_dot_pq2_0_q8_1`, `vec_dot_ptq1_0_q8_1(_multi)`.
- Decode math is small: PQ2_0 = Q2_0 codec + one scale per 128 (vs 64); PTQ1_0 = base-3 trit stream (5 trits/byte, `w=v*3; q=(w>>8)-1`), zero LUTs.
- Four-layer work plan defined: (1) wire types into SYCL, (2) quantizer, (3) MMVQ kernel optimization, (4) B2 loader/FWHT.

**B6. Spec document (05:34).** `sessions/INT2_KERNEL_SPEC_20260919.md` filed in the fork. Headline finding: *"the Doom strategy is already shipped, and that's the problem"* — the bottleneck is memory-pipeline structure (18-byte block strides defeating coalescing = prime suspect), not datatype. Ranked strategies: S1 vectorized-stream dp4a GEMV rewrite (chosen), S2 popcount/sign-mask ternary dot (fallback), S3 ESIMD unpack pipe (risk), S4 INT2-native IDPAS/XMX (parked).

**B7. Delegation attempt failed (05:40–06:02).** Phase 0 Task 1 dispatched to a subagent; it died on output-token exhaustion (reasoning consumed the entire budget across 4 continuations, zero visible output). Lesson: port work done in-session instead.

---

## PHASE C — PHASE 0: WIRE THE TYPES (Sep 19, 22:56–23:59)

**C1. "Yes, proceed" → in-session port (23:03).** 19 files changed in `/home/johnk/projects/llama.cpp-sycl/`:
- `GGML_TYPE_PQ2_0`=142 / `PTQ1_0`=143 mirrored from fork GGUF ids, `COUNT`=144, block structs + `QK/QI/QR` macros + static_asserts in `ggml-common.h`.
- SYCL kernels `vec_dot_pq2_0_q8_1` (per-32-chunk dp4a) and `vec_dot_ptq1_0_q8_1` (whole-block SWAR trit decode) in `vecdotq.hpp` — all `__byte_perm` selectors replaced with constant shifts, **bit-exact vs CUDA oracles @ 9a9394a** (3000 random blocks).
- Commits: `092e26116` (wire into ggml + SYCL MMVQ), `e66e5252f` (tensor-level quantizers + 1d fallback), `8347c5918` (ggml_validate_row_data rows).

**C2. GPU hygiene incident (23:49).** Level-set on the box: two test servers (PIDs 9052/19012, ports 51398/51399) were holding 8.9 GB on the B70; prod boot tasks had terminated (0x41306) and not restarted. Both test processes killed, GPU verified clear. (Adheres to the standing rule: kill only cand-build leftovers, never prod.)

---

## PHASE D — PHASE 0.5: BONSAI 2 RUNNING END-TO-END (Sep 20, 00:00–04:30)

**D1. Rebase + cherry-pick.** Fork rebased onto upstream master (851 commits: their SYCL FWHT, GDN, hadamard machinery). Cherry-picked **#29077** (QuentinDanblon's PQ2_0/PTQ1_0 + hadamard-folded-weights loader wiring) — block structs byte-identical, merged clean. Found Bonsai-2's FWHT block size 1024 missing from upstream's table (stopped at 1280 with 1024 skipped) → added `fwht<1024>` case (`05658dd0f`, Sep 21 01:36 commit, done during this window's build debugging).

**D2. First B2 bench ever on this card (04:01):**
```
qwen35 27B PQ2_0 - 2.13 bpw  6.70 GiB 26.90B  SYCL   pp512   909.22
qwen35 27B PQ2_0 - 2.13 bpw  6.70 GiB 26.90B  SYCL   tg128   13.74
build: 5854a59e2
```
Gen-1 regression check: 15.30 tg / 1175 pp — dead even with 15.28 baseline. Rebase cost nothing.

**D3. Attribution matrix (03:55, all same build/card):**

| config | tg128 | BW util | vs Q6_K |
|---|---|---|---|
| Q6_K (Qwen3.8) | 22.87 | **83%** (504 GB/s) | 100% |
| gen-1 Q2_g64 | 15.30 | 19% | 67% |
| gen-1 PQ2_0 | 14.74 | 17% | 64% |
| **Bonsai 2 PQ2_0** | 13.74 | 16% | 60% |

Nailed down: the 60%-BW goal is not fantasy (Q6_K proves the infrastructure can hit 83%); the bottleneck is the PQ2_0 decode kernel itself; the smoking gun — Q6_K moves 3× the bytes and still runs 1.66× faster.

**D4. User verdict (03:51):** *"13t/s is NOT performant... Goal is not met."* → Phase 2 kernel rewrite authorized ("Proceed").

---

## PHASE E — PHASE 2 KERNEL CAMPAIGN, PART 1: THE v3 ERA (Sep 20, 04:17–11:22)

**E1. v3 kernel (04:17–05:28).** Lane-strided byte-walk kernel: 32 lanes per 128-block, coalesced q8_1, SWAR decode. Commits `5fb9c904f`, `528f1ba38` (half→float conversions), `463a28cae` (sycl::fma ambiguity), `40c6ade42` (per-block scale d — fixed first garbage generations).

**E2. v3 measured 37.55 t/s on gen-1 PQ2_0 (05:00)** — 2.5× over the 14.74 baseline, ~54% of roofline. **But generation check = garbage** (`" the to to to to R to..."`). Honest caveat recorded at the time: the speed is real (traffic shape identical), but not usable; possibly fast *because* wrong.

**E3. The elimination grind (06:56–09:00).** Systematically ruled out, with hardware proof:
- File layout: dumped real GGUF bytes; code histogram clean ternary (36/29/35/0.3%); v1's int16 indexing mathematically reduces to v3's `byte=e/4, bits=(e%4)*2` — identical, proven over all 128 elements.
- Scale bug found+fixed (v3 hoisted block 0's `d` across the row) — still garbage.
- SWAR decode math verified over all 256 inputs + 50-trial block-level numpy simulation — exact match.
- Standalone micro-repro on the B70: first run 23/32 lanes wrong → **that was a harness bug** (forgot q8 chunk pointer offset). After fix: **v3 is 32/32 lanes bit-correct including butterfly reduce, on-device.**
- Suspect localized: dispatcher — single-col gen path routes to v3, multi-token path routes through generic template; v1's server binary never exercised v3's launch context.

**E4. Box outage (08:27).** John-PC-2024 dropped off the network mid-investigation (SSH/ping dead, router fine). Blocked until user confirmed reboot (05:49 message: "It rebooted and is back online").

**E5. THE ROOT CAUSE (11:22, solved and shipped).** The build defines **`-DGGML_SYCL_WARP_SIZE=16`** — not 32. v3 hardcoded 32-lane mapping (`ci = lane/8` covering all 4 q8_1 chunks); with 16 lanes it silently computed only chunks 0–1 of every block, skipping half the dot product. Compiles clean, micro-repro passes (harness compiled without the define), garbage in the real server. The generic template never had the bug because its indexing generalizes across warp size.
- Device-side printf proved unusable in DPC++ AOT kernels (variadic functions rejected) — lesson banked for later.
- Result pushed as `v3-clean`: **gen-1 control pp512 = 1129, tg128 = 15.15 (parity with 15.28 baseline); B2 PQ2_0 = 875 pp / 13.63 tg** (parity with 13.74). Correctness restored, zero speed.

**E6. User verdict (12:58):** *"13t/s is unacceptable. That isn't a fix, that's a joke."*

---

## PHASE F — PHASE 2 PART 2: STRUCTURE EXPERIMENTS (Sep 20, 13:10–15:10)

**F1. v4 kernel (13:14–13:30).** Q6_K-shaped 2-blocks/warp int32-word kernel (`7c80be633`, `e32319002`, `862d478aa`, `81bba47e4` — **fp scale was truncated to int, root cause of v4 garbage**, fixed).

**F2. SWAR vec_dot (13:58–14:51).** `e30cc46f6` — SWAR-vectorized `vec_dot_pq2_0_q8_1` eliminating the ALU-bound scalar 2-bit decode; `6f3b0f49e` routed through template; `69809a6ef` reverted gen dispatch to v4 keeping SWAR in template for bench.

**F3. The flat line (15:08).** Four genuinely different decode paths — v3, v4, SWAR-in-template, SWAR-in-v3 — all **tg128 = 13.63 ± 0.00**. Working state: 65.6 ms MSPT, correct generation, 0/64 harness mismatches.

**F4. Anomaly noted (22:52–22:55).** Three observations filed:
1. Identical tg across four implementations isn't convergence, it's evidence the bottleneck isn't in the rewritten code.
2. Compute math doesn't close: 6.7 GiB at 13.63 t/s = ~91 GB/s effective vs Q6_K's ~460 GB/s on the same box.
3. A `launch_mul_mat_vec_q_moe` PQ2_0 dispatch at `mmvq.cpp:3183` had never been touched — if B2 token-gen is dominated by expert matmuls through that path, that explains identical benches.
- **Additionally discovered: `bench_b70.ps1` had been pointing at `cand-build-stock\bin\llama-bench.exe` — the headline bench was measuring the wrong binary all night.**

---

## PHASE G — ISOLATION BENCH + THE v6 CHASE (Sep 20 22:52 – Sep 21 02:11)

**G1. Standalone kernel timing harness built** (`pq2_time.exe` / driver in `sessions/`, `which` export in `ggml_debug_pq2_0_run`): FFN-scale single matvec (17408×5120, batch 1) on real device.

**G2. v6 designed and landed (23:11–00:26).**
- `c26060984` **perf(v5): k-parallel PQ2_0 MMVQ — lane-per-block, zero 8× template redundancy, 1152B contiguous loads**
- `3e56f1c21` fix(v5b) row mapping for MMV_Y launch; `b25dd6dfc` duplicate launcher signature; `b7bd8ab50` trace print; `a9a2ada19` debug which=3→v5
- `60ae7c2f7` perf(v5d): padded trip count + unroll 4
- `a9cb4941f` **perf(v6): 2 rows/warp PQ2_0 MMVQ — halves redundant q8_1 L2 traffic via lane-pair y reuse**
- `754ff1bb1` route tg base fn through v6; `57a3aad44` route switch_ncols case 2 (MTP tg) through v6n<2>; `6aa2d87ed` GGML_SYCL_PQ2_V6=0 env fallback; `3e7084c4c` default back to template, v6 opt-in
- `2cec5fa56` GGML_SYCL_PQ2_NO_MMVQ=1 forces dequant+GEMM (bisect aid); `7f6c7d19b` harness link fix

**G3. Isolation result (00:16):** **v6 = 0.091 ms = 261 GB/s — 2.4× faster than template.** The L2-y-traffic hypothesis seemed vindicated.

**G4. But end-to-end: flat (02:11).** v6 wired into tg path → 13.5 tg128, no gain. Table recorded: isolated 2.4× vs e2e flat.

---

## PHASE H — THE CORRECTNESS SAGA (Sep 21, 00:00–09:44)

**H1. The bisect that changed everything (02:46–03:00).** Debug infrastructure built: `GGML_SYCL_PQ2_DUMP=1` dumps real PQ2_0 blocks + q8_1 + dst from the MMVQ path (`f1b480dbb`), full-metadata dump v2 (`dd4023e0c`), 320 x-blocks dump (`5fbf85f68`), per-lane partial dump `GGML_SYCL_PQ2_LANEDBG=1` (`b43922f2d`), SYCL static capture fix (`2fc61eb2d`).

**H2. Bisect verdict (03:00):**
| Test | Result |
|---|---|
| v6 isolated micro-bench | 0.091 ms vs template 0.22 — 2.4× faster |
| v6 in tg path | 13.5 tg128 — flat |
| B2 server gen (all MMVQ variants) | garbage (`intrintrintr...`) |
| B2 on **CPU backend** (−ngl 0) | **coherent** |
| B2 with `GGML_SYCL_PQ2_NO_MMVQ=1` (dequant+GEMM on GPU) | **perfect** ("Paris. The capital of Germany is Berlin...") |

**Root cause identified: EVERY PQ2_0 MMVQ kernel — template, SWAR, v5, v6 — computed wrong results on real model weights.** The harness's "0 mismatches" was GPU-vs-GPU (SWAR vs template) — they agreed *with each other* but both diverged from the CPU C++ reference. Random test data never triggered it; the real ternary weight distribution did.

**H3. First fix attempt: SWAR decode mask bug (03:54).** `a0b775a3c` — unmasked `(uint64_t)(code-1)`: for ternary code 0 (weight −1) this became 0xFFFF…FF, poisoning every higher byte of the expanded dp4a vector. Real ternary weights are full of 0-codes → garbage; random harness data rarely hit it. Fix: mask `& 0xFF` before widening shift. (Gateway blip interrupted here — build state on box unknown; resumed cleanly.)

**H4. Still wrong → byte-offset fix (05:16).** `8f198517d` — SWAR chunk byte offset `iqs*4 → iqs*8`. Followed by a debug-instrumentation barrage: one-shot printf (`cc5d7e405`, then dropped — printf unusable), per-chunk lanedbg dump (`0fd0c8aeb`), static fix (`52e1418db`), lanedbg2 v5-only (`801ba9f37`), raw x/y block byte dump from inside kernel (`0f1680d0d`), bit_cast fix (`021eeb87d`), step-by-step chunk0 recompute (`e8c3a174f`), probe relocation (`7720abb1f` — **a probe block had been wrongly injected at the top of mmvq.cpp; relocated into the v5 kernel body**), per-chunk probe (`420519bf9`).

**H5. Byte-order discovery (~05:30).** Kernel reads bytes bit-reversed per 32-bit word vs host order (host `48 24 a0 80` ↔ device dword `0x80a02448`). Per-chunk dot math verified to match emulation exactly for all four words/lanes (e.g. [-23, 86, 21, 60]) — arithmetic right, *inputs* wrong.

**H6. THE REAL ROOT CAUSE (07:18, commit `bc2fea754`).** **Double y-pointer advance in the v5/v6 call sites** — passed `&y[iby0 + c]` AND `iqs=c` into the swar vec_dot, which itself does `bq8_1 + iqs`. Every chunk after the first read Q8_1 block 2c instead of c. The SWAR math was bit-exact end-to-end; the dispatch fed it the wrong quantized activations. **Fix: pass `&y[iby0]`, keep `iqs=c`.** (FTZ corrupts float↔uint32 bit_cast in probes — noted as a debug pitfall.)

**H7. Verified clean (07:30–09:10).** `0a172c396` strips all lanedbg/chunkdbg probes. Real generation on B70: coherent English output. **Goal 1 (correct GPU MMVQ on B2) MET.** Measured on the fixed build: **13.3–13.5 t/s — identical to pre-fix 13.74.** The fix bought correctness, zero speed — confirming tg was never MMVQ-correctness-bound.

**H8. Honest goal accounting (09:10):** Goal 1 ✅ met; Goal 2 (48 t/s) ❌ not met, ~16% BW, ~28% of goal. Remaining gap needs kernel-structure work, not correctness chasing.

---

## PHASE I — Q6 LESSON MINING + LUT TRANSPLANT (Sep 21, 09:44–10:14)

**I1. OpenCode session mining (09:15 request).** SQL queries against the OpenCode session DB located the Q6 timing work: dozens of Q6 decode experiments, verdict "exhaustive lever sweep (all negative)" — Q6 was already at ~85% BW. **One big kernel win found: `vecdotq-lutswap-v2` (NVFP4, Sep 3)** — `dpct::byte_level_permute` (~20 ALU ops, 12× per 16 elements) swapped for a 16-entry table load → tg128 10.90→20.65 (**+89%**), decode BW 160→302 GB/s. Patch preserved at `~/llama-sandbox-backup/20260903-phase1/vecdotq-lutswap-v2.patch`.

**I2. Q6 prod-env transfer test (10:05).** Ran `bench_q6k_ctrl.bat` verbatim on B2: **13.36 t/s — identical to bare-env 13.74.** The prod flags Q6 benefited from are irrelevant to PQ2_0. Gap is inside the PQ2_0 vec_dot itself.

**I3. LUT transplant (10:14, `32043b02f`).** Built a 256-entry table (`pq2_lut256` in `ggml-common.h`): each byte of packed 2-bit codes → 4 precomputed int8 weights in one indexed load. New `vec_dot_pq2_0_q8_1_lut` in `vecdotq.hpp`, bit-exact vs SWAR, wired into v5 gen dispatch.

**I4. Result: no gain.** LUT = 13.28 tg128 vs 13.36 SWAR (pp 988). **Third decode variant with identical tg** — forced the conclusion that vec_dot decode cost is not the tg bottleneck at batch 1.

---

## PHASE J — FINAL RESOLUTION: ISOLATION NUMBERS + THE HARNESS BUG (Sep 21, 11:58–14:05)

**J1. Isolation bench at FFN scale (11:58, `pq2_time.exe`):**

| Kernel | Time | Weight BW |
|---|---|---|
| template | 0.217 ms | 109 GB/s |
| v4 | 0.139 ms | 170 GB/s |
| **v5** | **0.136 ms** | **175 GB/s (best correct)** |
| v5n2 | 0.171 ms | 138 GB/s |
| v6 | 0.083 ms | 281.6 GB/s |
| v7 (new) | 0.149 ms | 158.7 GB/s |

Ceiling math: template-rate matmul alone predicts 15 t/s e2e; v5-rate predicts 24.5; v6-rate predicts ~39.5. Measured e2e 13.3 matches the *template* rate almost exactly.

**J2. v6 wired into gen dispatch → 20.18 tg128 (+51%) → generation = garbage ("avoravoravor…").** The 20 t/s was fake.

**J3. v7 built as the control** — v5 body (1 row/warp, full-warp reduce, proven) with v6's 8-warp group launcher. Build error caught (`mul_mat_vec_pq2_0_v5` used before declaration — moved after v5's definition).

**J4. THE HARNESS BUG (14:05, commit `39ed01ced`).** Deep probe work (pre-reduce lane-16 marker, lane-0 marker `dst[1]=77+tmp*1e-6`) revealed the marker values could not have come from the kernel being tested. Source audit of `sessions/pq2_kernel_diff.cpp` found it: **`which_v` was computed but never passed to `run()` — every "variant vs template" phase actually ran which=0 (v3) vs template.** The entire "v6 zeroes odd rows" 8/64 signature was **v3's documented known bug**. Five build cycles chased a ghost.

**J5. True verdicts with the fixed harness:**
- v7 (which=7): **0/64 mismatches — correct**, 158.7 GB/s (slower than v5 → 8-warp grouping doesn't help).
- v6 (which=5): **8/64 mismatches — genuinely broken** (probes prove lanes 16–31 never execute their row path).
- **v6's 281.6 GB/s was a half-work artifact** — it computed only half the rows; per-row rate ≈ 140 GB/s, worse than v5. There was never a magic 2.1× kernel.

**J6. Final dispatch state (commit `c39cfcc27`):** gen dispatch = **v5+LUT**, verified correct, 13.4 t/s e2e. Decode-side optimization declared done: three correct implementations (v4/v5/v7) land within noise; v5 is the decode ceiling for this design.

**J7. Where the remaining gap lives (the handoff to next work):** v5-rate matmuls alone predict ~24.5 t/s; e2e is 13.4 → **~10 ms/token goes to non-matmul work** (attention, FWHT fold, sampling, other ops). That is the single biggest identified lever toward 24 t/s, and it's bigger than any remaining vec_dot rewrite.

---

## FINAL SCOREBOARD

| Milestone | Status |
|---|---|
| Physics/roofline established (608 GB/s, byte-lever thesis) | ✅ Sep 16 |
| PrismML fork recon + INT2 kernel spec | ✅ Sep 19 |
| PQ2_0/PTQ1_0 wired into SYCL, bit-exact vs CUDA oracle | ✅ Sep 19 |
| Bonsai-2 running end-to-end on B70 (first ever) | ✅ Sep 20 04:01 |
| Gen-1 Q2_g64 regression parity through all changes | ✅ maintained |
| GPU MMVQ correctness on B2 (real generations) | ✅ Sep 21 07:18 |
| 48 t/s performance goal | ❌ 13.4 t/s (~16% BW) |
| Isolation: v5 = 175 GB/s best-correct kernel; v6 exposed as half-work artifact | ✅ Sep 21 14:05 |
| Diff-harness which_v bug found+fixed | ✅ Sep 21 14:05 |

**Key artifacts:**
- `sessions/INT2_KERNEL_SPEC_20260919.md`, `sessions/PQ2_PHASE1_20260920.md` (fork docs)
- `sessions/pq2_kernel_diff.cpp` (fixed dual-kernel diff harness)
- `pq2_time.exe` + `ggml_debug_pq2_0_run` export (isolation bench)
- `GGML_SYCL_PQ2_DUMP/LANEDBG/NO_MMVQ/V6` debug env switches
- Commits of record: `092e26116` (wire), `5854a59e2` (B2 types+loader), `05658dd0f` (FWHT 1024), `bc2fea754` (double y-advance fix), `32043b02f` (LUT), `c39cfcc27` (v5+LUT dispatch), `39ed01ced` (harness fix + v7)

**Hard-won lessons (for whoever picks this up):**
1. A GPU-vs-GPU diff harness proves *agreement*, not *correctness* — always diff against the CPU scalar reference.
2. `which_v`-style parameter plumbing must be asserted used — five builds were wasted testing v3 while believing we tested v6.
3. `-DGGML_SYCL_WARP_SIZE=16` on this build means any hardcoded 32-lane mapping silently drops half the chunks.
4. Pass either the offset pointer OR the chunk index into a vec_dot — never both.
5. Isolated kernel speedups must be validated against work actually performed (v6's 2.4× was 0.5×-work).
6. Device-side printf/variadics are unusable in DPC++ AOT kernels; use USM debug buffers.
7. Bench scripts can silently point at the wrong binary — verify the exe path in every headline measurement.
