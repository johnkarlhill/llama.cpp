---
name: model-arch-fa-map
description: Source-verified map of llama.cpp architectures to their flash-attention quirks and which SYCL FA path (MKL/SDPA vs TILE) each hits — a testing checklist for PRs #25025 and #25874
metadata: 
  node_type: memory
  type: reference
  originSessionId: 0064f37a-aff4-46be-add6-ac4c5d6cf942
---

# Model Architecture → Flash-Attention Path Map (source-verified)

Built 2026-07-07 by reading `llama.cpp/src/llama-arch.cpp`, `llama-model.cpp`, `llama-hparams.cpp`, and `src/models/*.cpp`. Purpose: a **ready-to-go test checklist** for when we modify + test frankenmerge (#25312) — pick one model per row to exercise every gate path — and **reviewer ammunition** for #25025 (each gate clause maps 1:1 to a real model family).

## The key realization
Our MKL gate clauses are not arbitrary — **each one guards against exactly one real architecture family.** Five hazard classes, five clauses:

| Gate clause | Guards against | Only these arches trip it |
|---|---|---|
| `logit_softcap == 0` | attn logit softcapping | **Gemma2** (only arch that routes softcap through `ggml_flash_attn_ext`) |
| `Q->ne[0] == V->ne[0]` | MLA (K≠V head dims) | **DeepSeek2/V3/R1, DeepSeek3.2, GLM-DSA, Kimi-Linear** (K=576, V=512) |
| `!sinks` | attention sinks | **gpt-oss** (`openai-moe`, required) + **mimo2** (optional) |
| `head_dim % 64 == 0` | non-mult-of-64 head dim | **Phi-2 (80), Phi-3-mini (96)** — falls out of n_embd/n_head, not an explicit override |
| `max_bias == 0` | ALiBi positional bias | **BLOOM, Baichuan-13B, Refact, Jina-BERT-v2** (all 8.0), **MPT, JAIS** (from GGUF) |
| `gqa_ratio >= 2` | MHA quant crash | any MHA model (n_head_kv == n_head): Phi-2, older Llama, Baichuan, DeepSeek-v1 |

## Master map (FA-relevant hparams)

Head dims are generic: `n_embd_head = n_embd/n_head`, overridable by GGUF `attention.key_length`/`value_length` (`llama-model.cpp:1153-1180`). K==V for everything EXCEPT MLA.

| Family | head_dim K/V | GQA type | Quirk (→ gate) | Prefill path (K≥1024) |
|---|---|---|---|---|
| Llama 2/3, Mistral, Mixtral, Vicuna | default (128) | GQA (older MHA) | none | **MKL/SDPA/ONEDNN** ✓ |
| Llama4 | default | GQA | chunked SWA 8192 (mask only) | **MKL** ✓ |
| Qwen 2/2.5/3 (+MoE) | 128 | GQA | none | **MKL** ✓ (tested 100%) |
| Gemma 3 | 256 | GQA | SWA (mask), softcap=0 | **MKL** ✓ |
| Gemma 4 (26B/31B) | 256 local / **512 global** | GQA | 5:1 SWA:global (mask) | **MKL** ✓ (tested; why cap=512) |
| **Gemma 2** | 256 | GQA | **softcap ~50** + SWA | **TILE** ⚠ softcap |
| **DeepSeek2/V3/R1, 3.2** | **576 K / 512 V** | MLA (MQA-compressed) | **MLA** | **TILE** ⚠ K≠V |
| **GLM-DSA, Kimi-Linear** | **K≠V** (MLA) | MLA hybrid | **MLA** | **TILE** ⚠ K≠V |
| **gpt-oss (openai-moe)** | default (64) | GQA | **attention sinks** + SWA | **TILE** ⚠ sinks |
| **mimo2** | default | GQA | **sinks** (optional) + SWA | **TILE** ⚠ sinks (if present) |
| **Phi-2** | **80** | MHA | %64 **and** MHA | **TILE** ⚠ head_dim + gqa |
| **Phi-3-mini** | **96** | GQA/MHA | %64 (SWA force-disabled) | **TILE** ⚠ head_dim |
| **MPT, JAIS** | default | MHA/MQA | **ALiBi** (GGUF key) | **TILE** ⚠ max_bias |
| **BLOOM, Baichuan-13B, Refact, Jina-BERT-v2** | default | MHA/MQA | **ALiBi = 8.0** | **TILE** ⚠ max_bias |
| Falcon (MQA), Starcoder (MQA) | 64/default | MQA (n_head_kv=1) | none (MQA passes gqa≥2) | **MKL** ✓ |
| Command-R, Cohere2 (+MoE), Yi | default (128) | GQA | Cohere2 SWA (mask) | **MKL** ✓ |
| Grok | default | GQA | softcap 30 but via **out-scale tanh, NOT FA** | **MKL** ✓ (softcap doesn't reach FA) |
| Exaone4, SmallThinker, Step35, LFM2, Olmo2, etc. | default | GQA | SWA std (mask only) | **MKL** ✓ |

## Testing checklist — one model per path

To exercise every branch of the gate when testing frankenmerge, grab one from each group:

**MKL/SDPA/ONEDNN (should accelerate) — the ✓ rows:**
- GQA head_dim=128: any Qwen3 or Llama3 (already our daily drivers)
- head_dim=256 + 512 global: Gemma-4 (already tested)
- MQA edge: a Falcon or Starcoder (n_head_kv=1 — confirm gqa_ratio=n_head passes, doesn't crash like MHA)
- head_dim=64: gpt-oss WITHOUT sinks won't exist, but a 64-dim GQA model confirms the small-but-valid head path

**TILE fallback (must stay correct, must NOT crash) — the ⚠ rows, one per clause:**
- softcap → **Gemma 2** (9B/27B)
- MLA/K≠V → **DeepSeek-V2-Lite** (small, cheap to run) or DeepSeek-R1 distill
- sinks → **gpt-oss-20b**
- head_dim %64 → **Phi-2** (80) or **Phi-3-mini** (96)
- ALiBi → **MPT-7B** or **BLOOM** (smallest available)
- MHA → **Phi-2** (also covers MHA quant path)

Each ⚠ model is a regression target: run coherence + `test-backend-ops -o FLASH_ATTN_EXT`; confirm it silently falls to TILE and produces correct output, never garbage/crash.

## Caveats
- **Prefill only.** The gate needs Q≥32 AND K≥1024. Decode (Q=1) ALWAYS falls to VEC/TILE regardless of arch — MKL/SDPA are prefill accelerators. Don't expect decode to hit MKL.
- **SWA doesn't affect our gate.** Sliding-window is a mask shape; the mask is still present, so SWA models pass the `mask` check. It only matters that a mask exists.
- **Source vs empirical.** Gemma's 512 global-layer head_dim is our empirical GATE-DBG observation on gemma-4; the generic loader reports 256 (per-layer variation not obvious from the arch switch). Trust the running model's actual `Q->ne[0]`.
- **ONEDNN inherits this exact map** — it feeds the same shapes into oneDNN SDPA, so the same families accelerate and the same ⚠ families must fall to TILE. Plus ONEDNN's own BMG-only (Xe-HPG) ceiling — see [[xetla-fmha-notes]].

**Why:** Turns "does MKL/ONEDNN handle model X?" from a crash-discovery into a lookup, and gives #25025/#25874 reviewers a clause-by-clause justification. See [[project_tps_sycl_perf]] for current PR state, [[mkl-fa-workflow]] for how to run the tests.
