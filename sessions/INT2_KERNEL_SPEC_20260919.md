

## PHASE 0 WIRING COMPLETE (2026-09-19, Hermes direct — subagent path failed, done in-session)

All files touched (git status M): ggml.h (ids 142/143, COUNT 144), ggml-common.h (block structs + QK/QI/QR macros + static_asserts), ggml.c (type traits), ggml-quants.c/h (ref quant/dequant), ggml-sycl/vecdotq.hpp (vec_dot_pq2_0_q8_1 + vec_dot_ptq1_0_q8_1, byte_perm -> shift ports, bit-exact vs CUDA oracles 9a9394a), mmvq.cpp (mul_mat_vec_q + ncols + MOE dispatch, both switches), dequantize.hpp (dequantize_pq2_0/ptq1_0), getrows.cpp, convert.cpp (to_fp16/to_fp32), ggml-cpu/quants.c+h (generic vec_dots + quantize wrappers), ggml-cpu.c (traits), arch-fallback.h (7 arch blocks), llama.h (ftypes 141/142/143), llama-model-loader.cpp, llama-quant.cpp, tools/quantize.

### CRITICAL GOTCHA FIXED (do not regress): 128-wide blocks in MMVQ templates
launch_mul_mat_vec_q_moe / mul_mat_vec_q walk x[ibx] with stride sizeof(block_q_t) per qk elements. For 128-wide blocks qk MUST be 128 (QK_PQ2_0/QK_PTQ1_0), NOT 32. With qk=32 the weight pointer walks 4x too fast and 3/4 of every block is skipped silently (compiles fine, garbage output). Lane math that makes it work:
- PQ2_0: <QK_PQ2_0, QI_PQ2_0(4), block_pq2_0, VDR 1>: blocks_per_warp=(32+3)/4=8, i from lane/4 step 8, iqs=lane%4 in 0..3 → 4 lanes per block, one 32-elem chunk each.
- PTQ1_0: <QK_PTQ1_0, QI_PTQ1_0(4), block_ptq1_0, VDR 4>: blocks_per_warp=32, i from lane step 32, iqs=0 → one lane per block, whole-block call.

### Other verified subtleties
- vec_dot_pq2_0_q8_1 consumes ONE 32-elem chunk (iqs 0..3), reads bq8_1 + iqs — matches fork CUDA signature minus kbx.
- vec_dot_ptq1_0_q8_1 consumes the WHOLE 128 block (4 q8_1 chunks), SWAR decode byte-verified vs CUDA (3000 random blocks, 0 mismatches incl. qh tail).
- dequantize_pq2_0 (getrows/convert, qr=1 pair semantics): element e is byte e/4 bits 2*(e%4) — pair elements can straddle bytes (iqs%4==3), compute per-element.
- PTQ1_0 element order (dequant/vecdot/CPU): stage walk c=(32,16,8) over qs[24]: c=32 no-op; c=16 window bytes 0..15 (n=k/16,m=k%16, k 0..79); c=8 window bytes 16..23 (n=(k-80)/8, m=(k-80)%8, k 80..119); qh: element 120+2n+h = digit n of qh[h]. Digit = ((uint16)(uint8)(byte*pow3[n]) * 3) >> 8 - 1 (uint8 wrap intentional).
- PQ2_0 scale is fp16 d per 128; q8_1 side standard 4 chunks ds.
- dmmv NOT used (type not in ggml_sycl_supports_dmmv); mmq disabled globally; reorder-mmvq list does NOT include new types (intentional).
- CPU: generic vec_dots added (scalar), traits reference them via arch-fallback aliases (7 arch blocks, mirrors q2_0 which also has no x86 SIMD).
- Syntax: gcc -fsyntax-only clean on ggml.c, ggml-quants.c, ggml-cpu/quants.c, ggml-cpu.c (with -D_GNU_SOURCE; SCHED_BATCH needs it, pre-existing). SYCL TUs not compilable on WSL — Windows build (build_sycl.bat) is first real compile.

### Next (Phase 1)
Compile on Windows, run Ternary-Bonsai-27B gen-1 Q2_g64 (it's Q2_0 NOT PQ2_0 — it exercises stock path; PQ2_0 correctness test needs a PQ2_0 gguf: Bonsai-2 7.2GB staged file works once FWHT exists, so CPU-side round-trip tests first). Attribution matrix H1-H5 per spec §attribution.


## Phase 0 build + validation log (Sep 19, later)

- Windows build pipeline: WSL branch pushed to myfork (origin=sycl-onednn-fa-quants) -> Windows clone C:\llama.cpp-build-sycl\llama.cpp fetches -> worktree cand-int2 @ 007f5059c -> build dir cand-build-int2 (AOT bmg_g31, DNN=ON, F16=ON, icx/Ninja, -j16). BUILD_OK (llama-bench + llama-quantize).
- GOTCHA: worktree add FETCH_HEAD picked up a STALE FETCH_HEAD (landed on bacacfad8, a Windows-clone-local commit not even on the branch). Always checkout the exact commit SHA, never FETCH_HEAD.
- First quantize attempt: ggml_quantize_chunk had NO PQ2_0 case -> fell to default, assert 8005 (result==nrows*row_size) spam. Fixed: added quantize_pq2_0/quantize_ptq1_0 wrappers (mirror fork's) + chunk cases (c5f828d43).
- Second: ggml_validate_row_data missing type 142. Fixed with VALIDATE_ROW_DATA_D_F16_IMPL cases (daec267f5).
- llama-quantize needs vcvars+setvars on PATH (0xC0000135 DLL_NOT_FOUND otherwise).
- QUANT_OK: Qwen3.8-27B Q6_K (21.4 GB, 6.56 BPW) -> PQ2_0 smoke (8.36 GB, 2.44 BPW), 866/866 tensors, 39.7s.
- Now: llama-bench -m ...PQ2_0-smoke.gguf -p 512 -n 128 -ngl 99 -r 3 on B70. This exercises vec_dot_pq2_0_q8_1 + our mmvq dispatch (qk=128) at decode.
- Baselines: Q6_K 22.84 t/s tg128 (stock build); gen-1 Q2_g64 15.28 t/s (stock Q2_0 path).
