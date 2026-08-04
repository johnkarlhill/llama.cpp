# llama.cpp SYCL Flash Attention — Intel Arc GPU Development

## Project: SYCL Flash Attention for Intel Arc GPUs

**Owner:** @johnkarlhill (johnkarlhill/llama.cpp fork)
**Upstream:** ggml-org/llama.cpp
**GPU:** Intel Arc Pro B70 32GB (Battlemage BMG-G21) + Intel Arc B50 16GB (secondary)
**Platform:** Windows x64, SYCL/oneAPI + oneDNN
**Build machine:** John-PC-2024 (192.168.10.142) — the only machine with Intel GPUs

## Active Branch: `sycl-onednn-fa-quants` (PR #25874)

### PR Status
- **#25025** — MKL GEMM Flash Attention: ✅ **MERGED** 2026-07-31 (commit 9d9a6d29f)
- **#25874** — Extended oneDNN SDPA to Q4_0-Q8_0/F32 KV caches: **OPEN, merge-ready** (rebased onto post-MKL master 2026-08-01)
- **#25214** — GPU Heartbeat (`--gpu-heartbeat`): Open, needs SYCL-only scope rename
- **#25312** — HYBRID: CLOSED (folded into #25874)

### Dispatch Architecture (fattn.cpp priority order)
1. **ONEDNN (150)** — F16 KV, native oneDNN SDPA (from #25222)
2. **HYBRID/ONEDNN-dequant (150)** — Q4_0-Q8_0, F32 KV dequant→F16→SDPA (our code, #25874)
3. **MKL (300)** — GEMM-based FA, all types, handles ALiBi/softcap/no-mask
4. **TILE/VEC (200)** — Fallback for small K and decode

### Gate Conditions for HYBRID
- K >= 1024, Q batch >= 32 (prefill only)
- head_dim >= 32, F16 mask, no attention sinks
- max_bias == 0, logit_softcap == 0
- BMG-only arch gate

## Build (on John-PC-2024 via SSH)

Access the build machine:
```bash
ssh johnk@192.168.10.142
```

Build command (from VS oneAPI prompt or the build script):
```cmd
cd C:\llama.cpp-build-sycl\llama.cpp
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force
cmake --preset x64-windows-sycl-release -DGGML_SYCL_DNN=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=bmg_g21
cmake --build build-x64-windows-sycl-release --config Release -j 16 --target ggml-sycl
```

Quick incremental build:
```cmd
cmake --build build-x64-windows-sycl-release --config Release -j 16 --target ggml-sycl
```

**CRITICAL:** `-DGGML_SYCL_F16=ON` or MKL silently uses fp32 GEMM (2-3x slower, ~300 t/s).

**ITERATION vs PRODUCTION RULE (learned the hard way, 2026-08-03):**
- **Iterating = NO `GGML_SYCL_DEVICE_ARCH`.** AOT (ocloc) makes every incremental build take 5-10+ min on link. Iteration must be JIT: reconfigure without DEVICE_ARCH (or `build.bat`, which rmdirs and reconfigures JIT). JIT kernels compile at runtime; build is fast.
- **Production = `GGML_SYCL_DEVICE_ARCH=bmg_g21`.** AOT kernels baked in. Only for final/benchmark builds.
- `build_quick.bat` inherits whatever the CMakeCache has. If the cache is AOT (from a production build), build_quick is NOT quick. Check the cache first: `findstr GGML_SYCL_DEVICE_ARCH build-x64-windows-sycl-release\CMakeCache.txt`.
- `GGML_SYCL_AOT` is a **dead flag** — it exists in no CMake file. AOT is controlled solely by `GGML_SYCL_DEVICE_ARCH` (ggml-sycl/CMakeLists.txt:195).

### Build Scripts (consolidated in `C:\llama.cpp-build-sycl\build-scripts\`)
| Script | Type | What it does |
|---|---|---|
| `build_quick.bat` | Incremental (JIT or AOT, uses existing cache) | Env init + `cmake --build ... --target ggml-sycl`. Uses VS2022 vcvars64 to match current CMakeCache cl.exe. Fixes the old bug (setvars without vcvarsall). |
| `build.bat` | Clean full rebuild, JIT | rmdir build dir, reconfigure WITHOUT `GGML_SYCL_DEVICE_ARCH` (JIT kernels), full build. Fast iteration. |
| `scripts\run_server.bat` | Server launcher | Qwen27B, q8_0 KV, MTP, 147K ctx (moved from llama.cpp root). |
| `scripts\mkl-server.bat` | Server launcher (aliases) | Referenced by mkl-test.sh. Keep. |

Production AOT build (bmg_g21) is the manual configure one-liner from the build command block above, then `build_quick.bat` for incremental rebuilds against that cache. No separate script needed; the old `build_sycl.bat` is archived.

Archived/unused scripts live in `C:\llama.cpp-build-sycl\build-scripts\archive\` (build_tmp, _bld, _build_now, build_test, bench_jm, mkl-build, build_sycl, old .orig versions).

## Test Procedures (on John-PC-2024)

### 1. test-backend-ops (correctness gate — run before every push)
```cmd
build-x64-windows-sycl-release\bin\test-backend-ops.exe test -b SYCL0 -o FLASH_ATTN_EXT -j 1
```
Expect 3961/3961. `-j 1` forces single-thread so a crash's last log line identifies the exact failing case.

### 2. Multi-Turn Coherence (Gemma 26B, quantized KV cross-contamination test)
Model: gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf, 147K ctx
Two sequential prompts in same session:
1. "Write a comprehensive essay... Arctic Ocean" (long, fills KV cache)
2. "Where is the capital of the UK?"

**Pass:** Prompt #2 answers "London" — not Arctic Ocean text or garbled output.
**Fail:** ~0-3% draft acceptance, TG collapses to ~7.5 t/s (KV cache corruption from missing stream sync).

### 3. Full Performance Benchmark (Qwen 27B, 32K prefill)
Expected: 900+ t/s peak prefill, ~20 t/s TG with MTP, ~90% draft acceptance.

### 4. Multi-Turn Server Test (Qwen 27B, 6→1→5 prompt sequence)
Three prompts in single session: long stress test, short factual, long technical.
Verifies KV cache integrity through cache rotations and checkpoint recovery.

## Env Vars
- `GGML_SYCL_FA_ONEDNN=1` — default, enables HYBRID + ONEDNN paths
- `GGML_SYCL_FA_ONEDNN=0` — disables oneDNN-based paths, routes to MKL/TILE
- `GGML_SYCL_ENABLE_MKL_FA=1` — default, MKL GEMM path
- `GGML_SYCL_ENABLE_MKL_FA=0` — disable MKL, TILE/VEC only
- `GGML_SYCL_FA_DEBUG=1` — per-call dispatch logging
- `GGML_SYCL_MKL_FA_DIAG=1` — dump first 64 output floats
- `SYCL_CACHE_PERSISTENT=0`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`

## Key Source Files
- `ggml/src/ggml-sycl/fattn.cpp` — kernel dispatch (ONEDNN→MKL→TILE)
- `ggml/src/ggml-sycl/fattn-hybrid.cpp` — HYBRID: dequant→SDPA pipeline + build_sdpa()
- `ggml/src/ggml-sycl/fattn-hybrid.hpp` — SDPA partition + declarations
- `ggml/src/ggml-sycl/fattn-mkl.cpp` — MKL GEMM FA kernel
- `ggml/src/ggml-sycl/fattn-buffers.hpp` — persistent K/V buffers
- `ggml/src/ggml-sycl/convert.cpp` — dequantization implementations

## Model Aliases (on John-PC-2024)
| Alias | Model | Location |
|---|---|---|
| Qwen27B | Qwen3.6-27B-UD-Q5_K_XL.gguf | C:\llama.cpp-prod\models\B70\ |
| Qwen35B | Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf | C:\llama.cpp-prod\models\B70\ |
| Gemma26B | gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf | C:\llama.cpp-prod\models\B70\ |
| Gemma31B | gemma-4-31B-it-qat-UD-Q4_K_XL.gguf | C:\llama.cpp-prod\models\B70\ |

## PR Hygiene (llama.cpp specific — CRITICAL)

The ggml-gh-bot flags PRs for "predominantly AI-generated" content. Rules to avoid:

1. **User writes PR descriptions.** Draft code, but the PR body/title/commits must be in natural voice — shorter, less polished, conversational
2. **Shorter is better.** One-line commit messages preferred over multi-bullet descriptions
3. **Less polish.** No symmetric bullet lists, no 8-row test matrices with checkmarks. "Tested on 6 models, all pass" beats a markdown table
4. **User owns architecture.** User-authored reasoning is the strongest anti-slop signal
5. **No `Co-Authored-By` lines.** Don't confirm the bot's suspicion
6. **NEVER use em-dashes.** The em-dash (—) is a tell-tale AI signature. Use period, comma, or parentheses instead. This applies to ALL prose: PR text, commit messages, Discord replies, code comments. (Chat replies to the user are lower-stakes.)
7. **Default to user's voice** — direct, technical, conversational, not a polished whitepaper

## Pitfalls
- **GGML_SYCL_F16=OFF** causes MKL to silently use fp32 GEMM (looks like TILE speed). Always verify cmake cache.
- **sizeof(sycl::half)==2** on this platform. Read device fp16 as raw uint16_t + manual convert.
- **K->nb[2] = padded seq stride** (view into 2×seq buffer). The quant dequant stride path was broken by this before.
- **joint_matrix is DEAD on BMG** with oneAPI 2026.0 — all 5 tile configs throw exceptions. Use ESIMD xmx::dpas instead.
- **Pool alloc race**: ggml_sycl_pool_alloc frees GPU memory without stream sync. Added stream->wait() in fattn-hybrid.cpp.
- **Build happens ONLY on John-PC-2024** — this server (Dell) has no Intel GPU and no oneAPI toolchain.

## Reference Docs
See `docs/claude-memory/` for the full Claude Code memory files extracted from previous sessions:
- `project_tps_sycl_perf.md` — PR status and project state
- `mkl-fa-workflow.md` — Build/test/benchmark workflows
- `pre-commit-smoke-tests.md` — Full test procedures
- `ai-pr-hygiene.md` — PR style rules
- `xetla-fmha-notes.md` — Xe flash attention landscape, ESIMD viability
- `model-arch-fa-map.md` — Which model families hit which FA paths
- `future-optimization-targets.md` — Post-PR optimization roadmap
- `vec-tile-decode-investigation.md` — TILE vs VEC at decode

## Working with OpenCode

This project lives on the OpenCode server (192.168.10.172:4096). All code lives here.
When builds/tests are needed, SSH to John-PC-2024 (192.168.10.142) and run them there.
This server cannot build or test — it's for code analysis, editing, and orchestration only.
