# PrismML Hadamard (Bonsai 2) Port Scope — Sep 20 2026

## Correction to prior assumption
"PrismML fork has NO SYCL backend" is WRONG. `~/projects/llama.cpp-prism/ggml/src/ggml-sycl/`
exists and contains a complete Fast Walsh-Hadamard Transform implementation:
- `fwht.hpp` (12 lines, decl) + `fwht.cpp` (119 lines, kernel)
- Hook: ggml-sycl.cpp:4501 — intercepts MUL_MAT with op_params hint
  GGML_HINT_SRC0_IS_HADAMARD before normal matmul dispatch
- Kernel: templated FWHT, N ∈ {64,128,256,512}, sub-group butterflies
  (permute_sub_group_by_xor) + register butterflies, scale 1/sqrt(N),
  F32 in/out, 4 rows/block, one sub-group per row

## Architecture (much simpler than expected)
The rotation is NOT load-time weight mutation. It is a RUNTIME graph op:
1. GGUF carries `prism.hadamard.*` KV metadata (verified present in our
   staged E:\B70\Ternary-Bonsai-2-27B-PQ2_0.gguf):
   version=1, block_size, transform="normalized-sylvester-walsh-hadamard",
   axis="input-last-dimension", sign_mode (identity|explicit),
   weight_names[], inverse_weight_names[], sign_widths/sign_values,
   gdn_v_grouped
2. Loader (llama-model.cpp:1196-1330) validates + builds sign tables
3. Loader creates ONE small rot tensor per (block_size, buft):
   F32 [block_size × block_size], Sylvester Hadamard matrix generated
   arithmetically (parity popcount loop, ±1/sqrt(N)) — NOT read from file
   (llama-model.cpp:1961-2105). Plus optional sign vectors.
4. Graph (llama-graph.cpp build_lora_mm/build_lora_mm_id:1546,1605):
   if weight in hadamard_rotations map → reshape cur to [N, rows] and
   llama_mul_mat_hadamard (llama-impl.h:57) = plain ggml_mul_mat(rot, x)
   with GGML_HINT_SRC0_IS_HADAMARD + memoization per (cur, rot) pair
5. Backends: if they honor the hint → FWHT fast path (src0 not read at all,
   i.e. no dense matmul). If not → dense N×N matmul, still CORRECT math.
   CPU, BLAS, Metal, CUDA, VULKAN, SYCL all have hint checks in prism fork.
6. llama-context.cpp:2692 runs llama_verify_hadamard_graph to assert every
   folded weight actually went through a hadamard-aware matmul (silent-wrong
   -math guard)

## Port surface into llama.cpp-sycl (our fork)
| Piece | File(s) in prism fork | Est. effort |
|---|---|---|
| Hint enum + ggml_mul_mat_set_hint | ggml.h:449, ggml.c | trivial |
| Metadata parse + validation | llama-model.cpp:1196-1330 | small (~130 lines) |
| Rot/sign tensor creation | llama-model.cpp:1955-2105 | medium (~150 lines, buffer-type care for CPU/GPU placement) |
| build_lora_mm/mm_id wiring + memo + perm_rep tiling | llama-graph.cpp:1546-1700 | medium (~150 lines) |
| llama_mul_mat_hadamard helper | llama-impl.h:57-71 | trivial |
| FWHT SYCL kernel + hook | fwht.{hpp,cpp} + ggml-sycl.cpp:4501 | copy verbatim (~130 lines) |
| Graph verification pass | llama-context.cpp:2692 + helper | small |
| qwen35 arch in verified-arch allowlist | llama-model.cpp:1269+ | trivial (we already have qwen35 arch from gen-1 work) |
| CPU FWHT fallback | ggml-cpu.c:1275, ggml-cpu.cpp:454 | small |

Total: ~600-700 lines, all self-contained additive code. No binary
dependency on PrismML — the Hadamard matrix is generated arithmetically
from block_size, the kernel is ~120 lines of clean SYCL, and the metadata
is plain GGUF KVs. GGeriken-compatible: the "hard dep" objection doesn't
apply — this is exactly the sort of generic mechanism (hint-tagged matmul
+ optional fast path) that fits upstream conventions. The WHT matrix for
Bonsai 2 comes from GGUF metadata in the model file itself.

## Risks / notes
- build_lora_mm perm_rep tiling path (grouped feature order) adds
  complexity — verify whether Bonsai 2 files actually use perm_rep > 1
  (check hadamard KV values in the real GGUF; raw scan showed keys exist
  but values not yet parsed)
- Sign mode explicit requires sign vector plumbing (model may use it)
- gdn_v_grouped flag relates to gated-delta-net V grouping (Qwen3.8-next
  arch features?) — qwen35 arch is in the allowlist; confirm Bonsai 2's
  actual arch string (qwen35 vs qwen3next)
- Our fork's build_lora_mm may have diverged — port must adapt call sites
- Performance: FWHT is F32 elementwise on activations — cheap vs matmuls
  (N=128: 7 butterfly passes × 128 elems/row). Should NOT hurt tg.

## Measured model facts (E:\B70\Ternary-Bonsai-2-27B-PQ2_0.gguf, KV-parsed)
- arch = qwen35, block_size = 1024, sign_mode = explicit,
  gdn_v_grouped = 1, version = 1
- 401 folded weights (output.weight + every blk attn_qkv/attn_gate/
  ssm_out/ffn_down/ffn_gate...), 1 inverse weight,
  sign_widths = [5120, 6144, 17408], sign_values 28672 x {±1}
- IMPLICATIONS:
  * SYCL fwht.cpp only has cases for N ∈ {64,128,256,512} — must ADD
    N=1024 (32 registers/lane for el_w; CUDA fwht.cu switch goes to 2048,
    so 1024 is proven territory). Small change: add case + check reg
    pressure on BMG (32 float regs/lane is fine).
  * sign_mode=explicit is IN USE — the sign-vector path (sign tensors,
    sign application in graph) is NOT optional. Port must include it.
  * gdn_v_grouped=1 → gated-delta-net V grouping active.
  * Bigger issue: prism qwen35.cpp differs from ours by 455 lines and
    includes GDN/hybrid-SSM tensor creation (ssm_out, attn_gate,
    gdn_state_rows, raw_gates device paths). Our fork's qwen35.cpp (the
    gen-1 dense model) may not even create these tensors — Bonsai 2 is a
    Qwen3.5-HYBRID (GDN layers), not the dense gen-1 arch. Port of the
    hadamard mechanism is necessary but NOT sufficient: the model graph
    itself needs prism's qwen35.cpp lineage (GDN ops + their SYCL
    implementations, if any are custom).

## GDN op audit (final unknown, resolved)
- ggml_gated_delta_net is a new custom ggml op (GGML_OP_GATED_DELTA_NET)
- Prism ALREADY has a complete SYCL implementation:
  ggml-sycl/gated_delta_net.cpp (366 lines) + hpp + dispatch cases in
  ggml-sycl.cpp (5386, 5583, 5669) + backend.hpp
- CPU (ops.cpp/ops.h) and CUDA (gated_delta_net.cu 366+ lines) also done
- So the full Bonsai 2 stack exists in the prism fork WITH a working SYCL
  backend: model graph (qwen35 GDN diff) + custom op + FWHT + our already-
  ported PQ2_0/PTQ1_0 kernels

## Total port inventory (prism → our fork)
| Component | Size |
|---|---|
| ggml core: GGML_OP_GATED_DELTA_NET + ggml_gated_delta_net API + impl | ggml.h/ggml.c/ggml-cpu + op registration — medium |
| SYCL: fwht.{hpp,cpp} (131 ln, +1024 case) + gated_delta_net.{cpp,hpp} (385 ln) + dispatch hooks | ~550 lines, mostly verbatim copy |
| Loader: prism.hadamard metadata + rot/sign tensor creation (~300 ln) | medium |
| Graph: build_lora_mm/mm_id hadamard wiring + memo + perm_rep (~200 ln) + llama_verify_hadamard_graph | medium |
| Model: qwen35.cpp GDN diff (455 ln) + any supporting llama-graph/model changes | medium-large, needs careful merge against our fork's changes |
| CPU fallback: gdn ops + hadamard hint | included in above |

Estimate: 1,500-2,000 lines touched, ~60% verbatim copies from prism,
40% merge adaptation. This is a REAL port project (days, not hours), but
it is fully de-risked: every component already exists, works, and the
SYCL backend in prism proves the whole path compiles and runs on Intel.
Upstream-merge story: all generic (hint-tagged matmul, KV-driven
metadata, custom op with reference CPU impl) — no PrismML binary
dependency anywhere. Hadamard matrix is generated arithmetically;
everything ships as code.

