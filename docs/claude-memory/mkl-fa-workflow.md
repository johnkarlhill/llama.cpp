---
name: mkl-fa-workflow
description: "Build, test, and benchmark workflows for the SYCL MKL Flash Attention PR on llama.cpp"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0064f37a-aff4-46be-add6-ac4c5d6cf942
---

# MKL Flash Attention Workflow

## Project Layout
- **Build area**: `C:\llama.cpp-build-sycl\llama.cpp` (source + build dir)
- **Build output**: `C:\llama.cpp-build-sycl\llama.cpp\build-x64-windows-sycl-release\bin\llama-server.exe`
- **Scripts**: `C:\llama.cpp-build-sycl\scripts\` (mkl-build.bat, mkl-server.bat, mkl-test.sh)
- **Logs**: `C:\llama.cpp-build-sycl\logs\`
- **Daily driver**: `C:\llama.cpp-prod\` (models, prompts, benchmark results)
- **Models**: `C:\llama.cpp-prod\models\presets\`
- **Prompts**: `C:\llama.cpp-prod\docs\Prompts.txt`
- **Bench results**: `C:\llama.cpp-prod\docs\`

## GPU
Intel Arc Pro B70 (Battlemage BMG-G21, 32GB VRAM)

## Build
```bash
source /c/llama.cpp-build-sycl/scripts/mkl-test.sh
mkl-build              # incremental
mkl-build --configure  # full reconfigure + build
```
Under the hood: `MSYS_NO_PATHCONV=1 cmd.exe /c "C:\llama.cpp-build-sycl\scripts\mkl-build.bat"`

## Key Source Files
- `C:\llama.cpp-build-sycl\llama.cpp\ggml\src\ggml-sycl\fattn-mkl.cpp` — MKL FA kernel (GEMM-based)
- `C:\llama.cpp-build-sycl\llama.cpp\ggml\src\ggml-sycl\fattn-hybrid.cpp` — HYBRID FA (dequant→oneDNN SDPA)
- `C:\llama.cpp-build-sycl\llama.cpp\ggml\src\ggml-sycl\fattn-hybrid.hpp` — SDPA partition + build_sdpa() declaration
- `C:\llama.cpp-build-sycl\llama.cpp\ggml\src\ggml-sycl\fattn.cpp` — kernel dispatch (ONEDNN reserved, HYBRID, MKL, TILE/VEC)
- `C:\llama.cpp-build-sycl\llama.cpp\ggml\src\ggml-sycl\fattn-buffers.hpp` — persistent K/V buffers for HYBRID
- `C:\llama.cpp-build-sycl\llama.cpp\ggml\src\ggml-sycl\convert.cpp` — dequantization implementations

## Build
Primary: `C:\llama.cpp-build-sycl\llama.cpp\build.bat` (rmdir + cmake --preset x64-windows-sycl-release -DGGML_SYCL_DNN=ON + build)
Incremental: `C:\llama.cpp-build-sycl\build_quick.bat`
**CRITICAL:** `-DGGML_SYCL_F16=ON` required or MKL silently uses fp32 GEMM (2-3× slower, ~300 t/s)

## Environment Variables
- `GGML_SYCL_ENABLE_MKL_FA=1` (default, MKL enabled) / `0` (TILE fallback)
- `GGML_SYCL_FA_ONEDNN=1` (default, HYBRID enabled) / `0` (MKL-only)
- `GGML_SYCL_MKL_FA_DEBUG=1` — FA-DISP watchdog
- `GGML_SYCL_MKL_FA_DIAG=1` — FA-DIAG: dump first 64 output floats
- `SYCL_CACHE_PERSISTENT=0`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`

## Model Aliases
| Alias | Model File | Tokenizer |
|-------|-----------|-----------|
| Qwen27B | Qwen3.6-27B-UD-Q5_K_XL.gguf | Qwen/Qwen3.5-9B |
| Qwen35B | Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf | Qwen/Qwen3.5-9B |
| Gemma26B | gemma-4-26B-A4B-it-UD-Q5_K_M.gguf | google/gemma-4-26B-A4B-it |
| Gemma31B | gemma-4-31B-it-qat-UD-Q4_K_XL.gguf | google/gemma-4-31B-it |

## Common Commands
```bash
source /c/llama.cpp-build-sycl/scripts/mkl-test.sh

# Build
mkl-build

# Quick coherence check on one model
mkl-smoke Qwen27B

# Benchmark MKL vs TILE at 32K
mkl-compare Qwen27B 32768 512

# Full validation on all models
mkl-test-all
```

## Correctness verification (test-backend-ops)
The authoritative FA correctness check. Compares SYCL against a CPU reference.
```
# rebuild just the SYCL lib + test binary from a oneAPI prompt
cmake --build build-x64-windows-sycl-release --config Release -j 16 --target ggml-sycl test-backend-ops
set GGML_SYCL_ENABLE_MKL_FA=1
build-x64-windows-sycl-release\bin\test-backend-ops.exe test -o FLASH_ATTN_EXT -j 1 > out.txt 2>&1
```
- `-j 1` forces single-thread so a crash's last log line is the exact failing case (parallel workers otherwise scramble the crash location).
- Expect final `3641/3641 tests passed` + `Backend SYCL0: OK`. Grep for `FAIL` (watch for false matches on diagnostic strings containing "FAIL").
- FA test cases are in `tests/test-backend-ops.cpp`: `make_test_cases_eval()` (run by `test` mode) vs `make_test_cases_perf()` (perf mode only — cases added here are NOT correctness-checked).
- The big FA param loop iterates hsk {40,64,72,80,96,128,192,256,320,512,576}. K is a view into a 2×-seq buffer (simulates a real KV cache), so K->nb[2] = 2×seq stride — this is the "padded seq-view" layout that broke the quant dequant stride path.

## GATE-DBG diagnostic pattern (temp, for gate debugging)
When a real model unexpectedly hits TILE, add a one-shot `fprintf(stderr, ...)` in the gate dumping D/nq/nkv/gqa/mask/sinks/bias/softcap. `GGML_LOG_INFO` gets filtered; `fprintf(stderr)` always shows. Remove before commit.

## Critical Detail: `sizeof(sycl::half)`
oneAPI has `sizeof(sycl::half) == 2` on this platform (confirmed). Device fp16 is always 2 bytes. When reading device fp16 buffers, use raw `uint16_t` and manual fp16→f32 conversion, or `unsigned short`, to avoid host-side alignment issues.

## Fixed Bug: dst interleaved layout
The MKL normalize kernel was writing dst output in dense head-major format when llama.cpp expects interleaved `(query * n_heads + head) * DV`. Fixed in `mkl_fa_normalize_head`. This is the ONLY line that changed for the actual fix — everything else was debug diagnostics.

## Server Kill
```bash
taskkill //f //im llama-server.exe
```

**Why:** When `mkl-coherence` returns a response but not a good one, see [[mkl-fa-bug-dst-layout]].

**How to apply:** Source `mkl-test.sh` in bash, then use the short function names listed above. For new models, add entries to both `mkl-server.bat` (alias → path) and `mkl-test.sh` (tokenizer mapping).
