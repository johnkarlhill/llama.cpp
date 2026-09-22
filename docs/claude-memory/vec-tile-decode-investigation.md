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

## Test plan (NOT started yet — saved for next session)

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

## Why

Records a real, documented-in-the-codebase performance regression (quantized-KV decode forced onto slower VEC on BMG) plus the full A/B methodology, so we don't re-derive the gate's history or re-design the benchmark next session. See [[project_tps_sycl_perf]] for PR state, [[pre-commit-smoke-tests]] for the correctness battery, [[ai-pr-hygiene]] for how to present findings.
