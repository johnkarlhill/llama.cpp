---
name: tps-monitor-project
description: SYCL flash attention PRs for llama.cpp on Intel Arc Pro B70
metadata:
  node_type: memory
  type: project
  originSessionId: 0064f37a-aff4-46be-add6-ac4c5d6cf942
  modified: 2026-08-02T03:40:33.068Z
---

# SYCL Flash Attention PRs — Current State (2026-08-01)

User is @johnkarlhill, contributor to ggml-org/llama.cpp. GPU: Intel Arc Pro B70 (Battlemage BMG-G21, 32GB VRAM).

## Merged PRs

### #25025 — MKL GEMM Flash Attention ✅ MERGED 2026-07-31
- **Branch:** `sycl-mkl-flash-attn`
- **Status:** MERGED into master as `9d9a6d29f` (2026-07-31). ggerganov merged after arthw approved.
- **Merged on master after:** `9b2a08881`; master at merge tip was `6f3c0a790`.
- **Note:** PR #25874 was rebased onto the post-merge master on 2026-08-01 to pick up the MKL dispatch block in fattn.cpp.

## Active PRs

### #25874 — Extended oneDNN SDPA to Q4_0-Q8_0/F32 KV caches (OPEN, merge-ready)
- **Branch:** `sycl-onednn-fa-quants`
- **Status:** Open (ready for review), labeled merge-ready 2026-08-01. Rebased twice: onto `9b2a08881` (Jul ~30, after #25880) then onto `6f3c0a790` (Aug 1, after #25025 merge).
- **Parent:** #25222 (oneDNN SDPA F16 baseline, already merged into master)
- **Latest commit:** `d688c52a9` (2 commits: `ee36b4d36` main + `d688c52a9` SYCL.md). 2 files, 142 insertions.
- **Supported KV types:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (dequant via to_fp16_sycl / to_fp16_nc_sycl), F32 (cont_to_f16_sycl<float>)
- **Excluded:** BF16 (no strided conversion kernel), IQ types (model-weight-only quants)
- **Gate:** Non-F16 requires K>=1024 + Q>=32 (prefill only). F16 at any length. BMG-only arch gate.
- **Key fixes after rebase (from master's #25880):** scale uploaded via `single_task` kernel instead of async memcpy from stack local (root cause of the old "GGGGG" garbage); sync back to `device_count > 1` guard (no unconditional `wait_and_throw()`); `GGML_SYCL_FA_ONEDNN_MAX_KV` escape-hatch env var added.
- **Dispatch order in fattn.cpp:** ONEDNN (150) checked first, then MKL (300), then TILE (200). ONEDNN wins when both qualify.
- **Build:** `cmake --preset x64-windows-sycl-release -DGGML_SYCL_DNNL=ON -DGGML_SYCL_F16=ON`

### #25214 — GPU Heartbeat (`--gpu-heartbeat`)
- **Status:** Open, 1 approval (arthw) but blocked by 2 change requests
- **Key objections:** multi-backend scope (wants SYCL-only), need non-Intel reproducer
- **Next step:** Rename to SYCL-only (unblocked now that #25025 merged)

## Merged PRs
- **#25025** — MKL FA, merged 2026-07-31 as `9d9a6d29f`.
- **#25222** — oneDNN SDPA native F16 by @hmscider. Merged into master. Provides the SDPA partition builder and BMG gate that #25874 extends.

## Merge Strategy (updated)
1. ~~**#25025**~~ ✅ MERGED
2. **#25874** — OPEN, merge-ready (rebased onto post-MKL master 2026-08-01)
3. **#25214** last (heartbeat, needs SYCL-only scope change)

## Closed PRs
- **#25312** (HYBRID) — closed. All relevant code folded into #25874.

## Testing (see [[pre-commit-smoke-tests]] for full details)
- `test-backend-ops.exe -b SYCL0 -o FLASH_ATTN_EXT -j 1` — 3961/3961 for #25874
- Multi-turn coherence: Gemma 26B (Arctic Ocean → UK capital), Qwen 27B (6→1→5 prompt sequence)
- Qwen 27B 32K prefill: 900+ t/s peak, 800+ t/s at 32K, TG ~20 t/s with MTP

## Env Vars
- `GGML_SYCL_FA_ONEDNN=0` — disable ONEDNN SDPA path, routes to TILE
- `GGML_SYCL_ENABLE_MKL_FA=0` — disable MKL FA, routes to TILE/VEC
- `GGML_SYCL_FA_DEBUG=1` — per-call dispatch logging

## Build
- Incremental single target: `cmake --build build-x64-windows-sycl-release --config Release -j 16 --target ggml-sycl` (from oneAPI prompt)
- Shared-lib build: exe loads `ggml-sycl.dll` at runtime

## Key Pitfall
`GGML_SYCL_F16=OFF` causes MKL to silently use fp32 GEMM — looks like TILE speed (300 t/s). Always verify cmake cache.

## Follow-up Lead
Quantized-KV decode is forced onto VEC (slower on BMG); TILE wins +50-68% at 64K-118K. Full investigation + test plan in [[vec-tile-decode-investigation]]. Not started.

## Decode A/B Status (2026-08-04)
A/B executed: TILE wins every cell, no crossover (VEC 18.39→4.21 t/s vs TILE 19.79→7.80 t/s across 4K-118K, MTP-on 17.65→20.14 at 118K). Gate decision: quantized KV decode (Q<=2) → TILE unconditionally. Model×quant validation matrix pending. Full data in [[vec-tile-decode-investigation]].

**Why:** See [[mkl-fa-workflow]] for build/test commands. See [[pre-commit-smoke-tests]] for test procedures. See [[ai-pr-hygiene]] for PR-comment tone.
