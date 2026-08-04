---
name: vec-tile-decode-investigation
description: "Decode VEC-vs-TILE dispatch: TILE wins at BS=1 on BMG for quantized KV, gate provenance in CUDA/SYCL PRs, and the shape-matrix A/B test plan"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0064f37a-aff4-46be-add6-ac4c5d6cf942
  modified: 2026-08-02T03:40:07.863Z
---

# VEC vs TILE at decode — investigation + test plan (2026-08-01)

A Discord user (@johnkarlhill relayed, model `Laguna-S-2.1-UD-Q2_K_XL`, q8_0 KV, ctx 122880, B50+B70 layer split) found that force-routing decode (Q=1) to TILE instead of VEC improves TG substantially:

```
depth    stock     TILE
64k      6.78  ->  10.20 t/s   (+50%)
118k     4.42  ->   7.43 t/s   (+68%)
```

They report "VEC lost in every single case, no matter how deep the context was."

## The dispatch gate (fattn.cpp, `ggml_sycl_get_best_fattn_kernel`)

At decode (Q=1) with quantized KV, the code FORCES VEC with no escape hatch:

```cpp
if (can_use_vector_kernel) {
    if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        if (Q->ne[1] == 1) {
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_VEC;   // F16, Q=1, MHA -> VEC
            }
            // F16, Q=1, GQA -> FALLS THROUGH TO TILE (already!)
        }
    } else {
        if (Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;  // quantized, Q<=2 -> FORCED VEC
        }
    }
}
return BEST_FATTN_KERNEL_TILE;
```

F16+GQA already routes to TILE at Q=1; quantized never got the `gqa_opt_applies` check.

## Why TILE wins at BS=1 on BMG

- VEC: one work-group per (head, sequence) → very few work-groups, each sweeps the entire KV cache. GPU mostly idle.
- TILE: splits KV dimension into chunks → many concurrent work-groups per head. Better occupancy. At 122K ctx: VEC ~32-64 groups vs TILE ~61,000.
- Structural reason: TILE's multiple-warps-per-Q-column design (added for exactly this purpose — see provenance below).

## Provenance of the gate (source-verified)

- SYCL gate came from **PR #20190** ("[SYCL] supprt Flash Attention for fp32/fp16/Q4/Q5/Q8", merged 2026-03-08). Copied verbatim-ish from CUDA's `fattn.cu`. Never benchmarked on BMG.
- PR #20190's OWN numbers admit "TG is increased in more cases" — some decode regressed (`Qwen3.5-27B-Q2_K: -28.22%`, `Qwen3.5-35B-A3B: -13.44%`). Tested only on A770 + i7-13700K iGPU.
- **CUDA already knew the premise was wrong**: PR #16492 ("CUDA: faster tile FA") added parallel warps per Q column and wrote verbatim: "With this additional optimization the tile kernel seems to now be a better choice for batch size 1 than the vector kernel, particularly for AMD hardware."
- CUDA's current gate has per-arch branches (`cc >= GGML_CUDA_CC_ADA_LOVELACE`) AND a shape heuristic: `!(gqa_ratio > 4 && K->ne[1] >= 8192)`. The SYCL port never got this arch/shape tuning.
- **Not a pure arch question**: current code already routes F16+GQA+Q=1 → TILE. So it's also model-shape-dependent (gqa_ratio, head_dim, context depth). A naive `arch==BMG` gate risks regressing shapes where VEC genuinely wins.

## Test plan (EXECUTED 2026-08-04 — results below)

1. **Test-only patch** in `fattn.cpp` right after `if(!g_ggml_sycl_enable_flash_attention) return BEST_FATTN_KERNEL_NONE;`:
   ```cpp
   static int fa_force_decode = ggml_sycl_get_env("GGML_SYCL_FA_FORCE", 0);
   if (fa_force_decode != 0 && Q->ne[1] <= 2) {
       return fa_force_decode == 1 ? BEST_FATTN_KERNEL_VEC : BEST_FATTN_KERNEL_TILE;
   }
   ```
   REVERT before pushing to PR branch.
2. Build ggml-sycl + llama-bench + llama-cli.
3. **Correctness gate first**: existing coherence tests (Gemma Arctic Ocean, Qwen 6→1→5) with FORCE=1 AND FORCE=2, confirm both kernels coherent multi-turn before trusting t/s.
4. **A/B script** (full `decode-ab.sh` drafted in this session's transcript): llama-bench `-p <ctx> -n 64 -c ctx+512 -fa -r 3 --cache-type-k/v q8_0`, sweep modes auto/vec/tile × ctxs 4096,16384,32768,65536,98304. Set `GGML_SYCL_FA_ONEDNN=0 GGML_SYCL_ENABLE_MKL_FA=0` to isolate VEC/TILE. `auto` = control (must match vec on quantized KV, tile on F16+GQA).
5. **Shape matrix** (each × q8_0 + f16 KV): MHA (gqa=1), GQA-2/4, GQA-8 (Qwen3.6-27B), Gemma (hd 256/512 + global layers), and the Discord repro cell (Laguna UD Q2_K_XL, 64K + 118K) first.
6. **Read the crossover**: plot t/s vs ctx. Flat TILE win → pure arch gate ok. Crossover → need context term like `K->ne[1] >= <crossover>` (mirror CUDA's `gqa_ratio > 4 && K->ne[1] >= 8192`). VEC wins somewhere → gate must be narrower.
7. If confirmed, propose the fix on a llama.cpp issue/PR with the Discord repro + matrix.

## RESULTS (2026-08-04) — TILE wins every cell, no crossover

**Setup:** ThinkingCap-Qwen3.6-27B-Q6_K, q8_0 K/V cache, AOT production build (bmg_g21), llama-benchy via server, `--no-adapt-prompt --runs 1`, MTP OFF. FORCE=1 (VEC) vs FORCE=2 (TILE). Prefill identical between modes (865 vs 847 at pp4096, through 694.41 vs 694.35 at pp118784) — FORCE patch confirmed decode-only.

| Depth | VEC tg t/s | TILE tg t/s | Delta |
|---|---|---|---|
| 4096 | 18.39 | 19.79 | +7.6% |
| 16384 | 13.50 | 17.05 | +26.3% |
| 32768 | 10.00 | 14.28 | +42.8% |
| 65536 | 6.58 | 10.84 | +64.7% |
| 98304 | 4.89 | 8.75 | +78.9% |
| 118784 | 4.21 | 7.80 | +85.3% |

**MTP-on cell (production config):** 118K depth, draft-mtp (q8_0 draft KV): VEC 17.65, TILE 20.14 = **+14.1%**. Delta survives MTP at depth because the MTP draft is the same model: its single-token draft decode is itself a Q=1 quantized-KV attention call, hitting the same VEC/TILE fork. (Earlier 4K MTP parity, 38.38 vs 37.85, was a shallow-context artifact.)

**MTP prefill tax (side observation):** MTP costs ~27% of prefill from 0 context (1300 → 950 t/s start). Separate from decode question.

**Verdict:** flat TILE win, monotonic, no crossover in the tested envelope. The CUDA shape heuristic (`gqa_ratio > 4 && K->ne[1] >= 8192`) is NOT needed on BMG — this is a per-arch structural result (occupancy), not shape-dependent. Decode t/s is bandwidth-bound; VEC's ~32-64 work-groups vs TILE's ~61,000 at depth decides it.

**Gate decision:** quantized KV decode (Q<=2) → TILE unconditionally. Remove forced VEC in the quantized branch (fattn.cpp:265-267); fall-through already lands on TILE. F16 path unchanged (no F16 data yet; F16 GQA already routes TILE at Q=1, F16 MHA still VEC).

**Methodology notes:**
- MTP must be OFF to see the full delta; MTP masks it at shallow depth
- `--no-adapt-prompt` required for identical pp counts between runs
- Discord repro confirmed: Laguna Q2_K_XL q8_0 KV, 64K/118K, "VEC lost in every case" (+50%/+68%)

## Follow-up: model × quant matrix (NEXT)

Validate the delta holds across architectures and KV quants before PR:

- **Models:** Qwen3.6-35B-A3B (MoE A3B, Q4_K_XL), Gemma 4 12B (dense, interleaved), Gemma 4 26B A4B (MoE A4B, hd 256/512 global)
- **KV quants:** q4_0, q8_0, f16 — per model where supported
- **Depths:** 32K and 118K cells (the extremes)
- Same llama-benchy protocol: `--no-adapt-prompt --runs 1`, FORCE=1 vs FORCE=2, MTP off for the matrix, one MTP-on spot check

**If delta holds across the matrix → PR is a switch:** quantized KV decode routes to TILE by default, with an env switch (e.g. `GGML_SYCL_FA_DECODE_KERNEL=vec|tile|auto`) for escape hatch, mirroring how `GGML_SYCL_ENABLE_MKL_FA` works. Not a shape heuristic — a flat per-arch default.

## Why

Records a real, documented-in-the-codebase performance regression (quantized-KV decode forced onto slower VEC on BMG) plus the full A/B methodology, so we don't re-derive the gate's history or re-design the benchmark next session. See [[project_tps_sycl_perf]] for PR state, [[pre-commit-smoke-tests]] for the correctness battery, [[ai-pr-hygiene]] for how to present findings.
