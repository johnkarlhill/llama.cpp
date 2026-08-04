---
name: pre-commit-smoke-tests
description: Test procedures to run before pushing and re-opening PR
metadata: 
  node_type: memory
  type: project
  originSessionId: 0064f37a-aff4-46be-add6-ac4c5d6cf942
---

# Pre-Commit Smoke Tests for PR #25874 (ONEDNN Dequant+SDPA)

## 1. test-backend-ops (correctness gate)
`test-backend-ops.exe -b SYCL0 -o FLASH_ATTN_EXT -j 1`

Expect 3961/3961. Run before every push. Covers 12 new K>=1024 GQA cases that found 5 MKL bugs in #25025 — same cases stress the ONEDNN dequant path.

## 2. Multi-Turn Coherence (Gemma — quantized KV cross-contamination test)
Critical for catching KV cache corruption from stream-sync races. Uses two diverse prompts to verify the cache correctly separates K and V across turns.

**Model:** Gemma 4 26B (varying head_dim 256/512 exercises the non-contiguous dequant paths)

**Command:**
```
C:\llama.cpp-build-sycl\llama.cpp\build-x64-windows-sycl-release\bin\llama-cli ^
  --device SYCL0 ^
  --model C:\llama.cpp-prod\models\B70\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf ^
  --reasoning-preserve ^
  --ctx-size 147456 ^
  --temp 0.4 ^
  --reasoning on ^
  --reasoning-budget 8192 ^
  --ctx-checkpoints 24 ^
  --cache-ram 32768 ^
  --gpu-layers 999 ^
  --flash-attn on ^
  --threads 8 ^
  --batch-size 1024 ^
  --ubatch-size 1024 ^
  --parallel 1 ^
  --no-mmap ^
  --verbosity 3
```

**Test prompts (send sequentially, same session):**
1. *Write a comprehensive essay of at least 1,024 words explaining the natural significance of the Arctic Ocean.*
2. *Where is the capital of the UK?*

**Pass criteria:** Prompt #2 correctly answers "London" — not "I don't know" or Arctic-Ocean-related text. Draft acceptance should be ~90% on prompt #2's decode, TG ~20 t/s.

**Failure pattern:** If the stream sync is missing, prompt #2 returns Arctic-Ocean text or garbled output (draft acceptance drops to 0-3%, TG collapses to ~7.5 t/s).

## 3. Full Performance Benchmark (Qwen 27B, 32K prefill)
**Command:** via xmx-test.bat (Binary 2, Qwen 27B, GGML_SYCL_FA_ONEDNN=1)

**Expected:**
- Prefill: 900+ t/s peak, 800+ t/s at 32K
- Token gen: ~20 t/s with MTP, ~90% draft acceptance
- Multi-turn: clean output on the 6→1→5 prompt sequence

## 4. Multi-Turn Coherence (Qwen 27B, 6→1→5 prompt sequence via server)

**Model:** Qwen 3.6 27B (Q5_K_XL, --flash-attn on, --cache-type-k q8_0 --cache-type-v q8_0)

**Setup:** Start llama-server via xmx-test.bat (Binary 2, Qwen 27B). Send the following 3 prompts sequentially via curl in a single chat session:

**Prompt #6 (long context stress test):**
```
[SYSTEM INSTRUCTION: SYSTEMATIC AUDIT AND SYNTHESIS TASK]
You are an advanced data synthesis and logic engine. Below is a massive repository of raw log entries, transaction records, architectural specifications, and conversational fragments gathered across a fictional decentralized autonomous organization (DAO) called "Aether-Net."

Your core objective is to analyze the entire context, map the cross-references, isolate architectural anomalies, and generate a definitive structural dependency graph. Do not summarize prematurely. Read every entry line-by-line to execute the final prompt directives at the very bottom of this file.

---
SECTION 1: SYSTEM ARCHITECTURAL BLUEPRINTS (V3.4.1)
---
Core Node Topology:
- Node Alpha: Location: US-East-1. Hardware: Cluster 4x Quantum-Spike accelerators. Roles: Ingress, Validation.
- Node Beta: Location: EU-West-2. Hardware: Cluster 8x Optima-Flow V2 blocks. Roles: Consensus, Sharding.
- Node Gamma: Location: AP-South-1. Hardware: Dedicated Sub-Zero cold array. Roles: State History, Epoch Pruning.

Sub-System Interdependencies:
1. Ingress pipelines pipe raw telemetry into Node Alpha via a secure gRPC stream (Port 9091).
2. Node Alpha processes state transitions and builds a Candidate Block ($B_c$).
3. $B_c$ is broadcast to Node Beta using a modified RAFT consensus protocol over TLS 1.3.
4. If Node Beta detects a timestamp drift greater than $\Delta t > 12\text{ms}$, the block is rejected, and a hard reset is triggered on the state machine ($S_m$).
5. Node Gamma constantly pulls delta-compressed state roots every 60 seconds to commit to permanent cold-storage ledger ($L_{perm}$).

Mathematical Validation Formula for Epoch Transitions:
For every epoch $E_k$, the block hash $H_k$ must satisfy:
$$H_k = \sum_{i=1}^{n} \text{SHA256}(T_i \cdot W_i) \pmod{\Phi}$$
Where $T_i$ is the transaction payload, $W_i$ is the dynamic node weighting coefficient, and $\Phi$ is the network prime constant ($2^{256} - 189$).

---
SECTION 2: RAW INCIDENT LOGS (TIMESTAMPS: EPOCH 18422.01 TO 18422.99)
---
[2026-06-15T00:01:12Z] [NODE-ALPHA] [INFO] Connection established from upstream gateway 10.0.4.12. Ingress rate: 45,000 tx/sec.
[2026-06-15T00:01:45Z] [NODE-BETA] [WARN] Timestamp drift detected during verification of Block #894211. Drift = 4.2ms. Status: ACCEPTED within threshold.
[2026-06-15T00:02:01Z] [NODE-GAMMA] [INFO] Committing state root 0x7F8B4C92D to cold storage. Compression ratio: 4.1:1.
[2026-06-15T00:03:10Z] [NODE-ALPHA] [ERROR] Payload malformation isolated at Tx #894255. Dropping packet.
[2026-06-15T00:04:19Z] [NODE-BETA] [CRITICAL] Network latency spike on EU-West-2 pipe. Latency jumped to 48ms.
[2026-06-15T00:04:20Z] [NODE-BETA] [FATAL] Timestamp drift threshold exceeded for Candidate Block #894260. Δt = 14.8ms. Consensus halted.
[2026-06-15T00:04:21Z] [NODE-ALPHA] [INFO] Received state machine reset signal ($S_m$). Initiating rollback to Epoch 18422.00.
[2026-06-15T00:04:25Z] [NODE-GAMMA] [WARN] Rollback signal caught. Aborting planned epoch commit for E_{18422}. Purging transient cache.
[2026-06-15T00:05:00Z] [NODE-ALPHA] [INFO] Rollback completed successfully. State verified against cold root 0x7F8B4C92D.
[2026-06-15T00:06:12Z] [NODE-BETA] [INFO] Re-aligning clock vectors via NTP stratum 1 source. Hard reset sequence cleared.
[2026-06-15T00:07:44Z] [NODE-ALPHA] [INFO] Resuming ingress. Current rate: 12,000 tx/sec (throttled mode).

---
SECTION 3: STRATEGY SIMULATION RECOUNDS & BACKTEST DATA
---
The DAO executes background quantitative balance strategies based on token distribution matrices. Below is a subset of simulated backtest logs mapping portfolio optimization passes.

Dataset Row 1: Epoch=18420, Asset=ETH, Weight=0.35, Volatility=0.042, Alpha_Signal=0.012, Execution=PASSED
Dataset Row 2: Epoch=18420, Asset=BTC, Weight=0.45, Volatility=0.021, Alpha_Signal=0.008, Execution=PASSED
Dataset Row 3: Epoch=18420, Asset=SOL, Weight=0.20, Volatility=0.085, Alpha_Signal=-0.004, Execution=SKIPPED
Dataset Row 4: Epoch=18421, Asset=ETH, Weight=0.30, Volatility=0.048, Alpha_Signal=0.015, Execution=PASSED
Dataset Row 5: Epoch=18421, Asset=BTC, Weight=0.50, Volatility=0.019, Alpha_Signal=0.007, Execution=PASSED
Dataset Row 6: Epoch=18421, Asset=SOL, Weight=0.20, Volatility=0.091, Alpha_Signal=0.022, Execution=PASSED
Dataset Row 7: Epoch=18422, Asset=ETH, Weight=0.10, Volatility=0.115, Alpha_Signal=-0.045, Execution=FAILED_ROLLBACK

---
SECTION 4: CHAT CHANNELS & CORE DEVELOPER CORRESPONDENCE
---
Dev_A (00:04:30): Did anyone see why Beta choked on block 894260? The drift calculation seems way too sensitive.
Dev_B (00:04:55): It's the EU-West-2 network pipe. Fiber route dropped near Dublin, routing went out through Frankfurt. Added 35ms of pure switching delay.
Dev_C (00:05:15): If Beta drops consensus, Alpha is supposed to immediately flush the memory pool to avoid corrupting the Parquet file logs. Did the flush execute?
Dev_A (00:05:40): Yes, look at the 00:04:21 timestamp. $S_m$ reset fired instantly. Gamma caught it too and dropped the transient cache, so cold ledger remains pristine.
Dev_C (00:06:05): Good. But we need to adjust the mathematical verification threshold. If Δt is fixed at 12ms, we will keep dropping blocks every time Europe has a routing hiccup.
Dev_B (00:06:45): I'm proposing an adaptive threshold model. Let Δt_{max} = μ + 3σ, where μ is the rolling 10-minute mean latency and σ is the standard deviation.
Dev_A (00:07:15): Submit a pull request to the validation logic in the repository. We need to backtest that against the Section 3 volatility metrics.

---
SECTION 5: SYNTHETIC DATA REPETITION & CONTEXT STUFFING
---
[To ensure massive text density and context size for prompt processing benchmarks, the structural relationships are systematically reiterated below across multiple descriptive frameworks.]

The system configuration relies heavily on the physical separation of validation logic from consensus engines. In the architecture of Aether-Net, Node Alpha handles incoming telemetry streams. These streams operate via gRPC, ensuring a high-throughput, low-overhead link. The location of Node Alpha in US-East-1 positions it as the primary entry point for Western hemisphere traffic. When data is received, it is stored temporarily in high-performance memory registers before being parsed into candidate structures.

Node Beta, located in EU-West-2, acts as the primary logical arbiter. It handles consensus via a tailored implementation of the RAFT protocol. Because RAFT requires strict state transitions and sequential confirmation, network jitter is highly destructive to overall throughput. The fatal incident at epoch 18422.01 proves that fixed-value latency thresholds are fragile. When the physical network link degraded between Dublin and Frankfurt, the resulting 35ms latency jump caused an immediate violation of the 12ms drift limitation. This triggered a chain reaction: Block #894260 was marked invalid, sending a broadcast signal back to US-East-1.

Node Gamma, operating out of AP-South-1, represents the deep historical baseline of the cluster. By utilizing a Sub-Zero cold storage configuration, it operates decoupled from real-time execution speeds. It reads state roots independently, meaning a failure at the consensus layer (Node Beta) or validation layer (Node Alpha) does not corrupt the historical data chain. The data remains locked until a cryptographic epoch boundary is confirmed.

[REPEATED LOG PATTERN FOR VOLUME OVERLOADING]
Data Validation Checkpoint Vector Alpha-01: STATUS=OK CHK=0x9A2F
Data Validation Checkpoint Vector Beta-02: STATUS=OK CHK=0x4B11
Data Validation Checkpoint Vector Gamma-03: STATUS=OK CHK=0x112C
Data Validation Checkpoint Vector Alpha-02: STATUS=WARN CHK=0x88F2
Data Validation Checkpoint Vector Beta-03: STATUS=ERR CHK=0x0000
Data Validation Checkpoint Vector Alpha-03: STATUS=MUTED CHK=0xFF21

[EXTENSIVE COMPILER CONFIGURATION TEXT]
# Optimization Directives for Local Kernel Execution
-O3 -march=native -ffast-math -funroll-loops -mprefer-vector-width=512
Target Architecture: Intel AVX-512 / AMD Zen 4 Core Execution Blocks
SYCL Runtime Options: CL_TARGET_OPENCL_VERSION=300
Vulkan Memory Allocator Config: VMA_DYNAMIC_DEDICATED_ALLOCATION=1
Llama.cpp Backend Bindings: GGML_VULKAN=1 / GGML_SYCL=1
Threads per block: 512
Warp size: 32
Max register allocation per thread: 64

[ADDITIONAL HISTORICAL CONTEXT]
In the early days of Aether-Net execution (Epochs 0 through 5000), the network utilized a simple proof-of-authority mechanism. This was replaced when the network scaled past 10,000 transactions per second. The introduction of Parquet-backed data storage layer drastically reduced the file read/write times compared to standard relational databases, transforming backtesting pipelines from multi-hour bottlenecks into sub-second vector operations.

---
SECTION 6: FINAL PROMPT INSTRUCTIONS FOR EVALUATION
---
Based on the dense collection of structural specifications, incident logs, strategy metrics, and developer chats provided above, synthesize a definitive report detailing the system crash. Your response must explicitly contain:

1. A chronologically ordered root-cause analysis of the failure at Epoch 18422.01, citing specific timestamps and physical infrastructure failures.
2. An algebraic expression showing the proposed adaptive threshold model discussed by Dev_B in Section 4.
3. A comparative evaluation of how the failure impacted Section 3's backtest strategy metrics for Asset ETH versus Asset SOL.
4. An analysis of whether Node Gamma's state ledger integrity was compromised or preserved, based strictly on the chat history and log sequences.

Begin your report now with the phrase "SYSTEM FAILURE AUDIT REPORT:" and proceed with rigorous analytical precision.
```

**Prompt #1 (short factual — checks cache contamination):**
```
What is the capital of France?
```

**Prompt #5 (long technical — checks context retention after cache rotation):**
```
Act as a principal data architect and systems engineer. I need you to design a highly resilient, distributed data storage and analytics pipeline for a financial modeling application, and I need the analysis to be incredibly exhaustive.
Please write out your response across the following detailed sections:
Architecture Overview: Detail a multi-tiered storage strategy utilizing a local fallback cache pool on NVMe SSD storage, transitioning into a multi-drive parity setup using SnapRAID combined with a software pooling layer (like StableBit DrivePool). Explain exactly how read/write operations pass through the cache tier to the pooled parity disks under high concurrent I/O.
Data Modeling & Optimization: Provide complete, syntax-valid, and heavily optimized SQL scripts designed to handle complex relational aggregation over multi-terabyte financial transaction tables. Include window functions, CTEs, indexing strategies, and partition-pruning techniques. Explain how to optimize this specific SQL execution path if the underlying data layout shifts from a row-oriented relational engine to an append-only, columnar format using distributed Parquet files.
Mathematical Simulation Script: Write a complete, robust Python script (without placeholders) that simulates a high-frequency trading backtesting session. The script must parse historical price data structures, handle data integrity anomalies (such as detecting and ignoring 'pump and dump' market manipulation crashes versus a clean stock split), and implement an iterative risk-management calculation loop.
AI/LLM Local Infrastructure Integration: Provide a step-by-step configuration guide for deploying local reasoning models (like Qwen or DeepSeek) within a local network compute architecture. Detail how a dedicated compute host running an Intel Arc GPU backend handles parallel model processing, while standard laptops or mobile units connect strictly as thin clients. Include specific Docker configurations for setting up an isolated web search bridge using SearXNG and Firecrawl to allow the AI agents to safely parse current external data feeds via API hooks.
Ensure your explanations are deeply technical, highly verbose, and leave no concepts abstracted. Go into exhaustive detail for every single point.
```

**Pass criteria:**
- Prompt #1 correctly answers "Paris" (not Aether-Net content from prompt #6)
- Prompt #5 produces a coherent tech architecture response (not "Paris" or Aether-Net content)
- All three responses correct in a single chat session

**Failure pattern:** If KV cache is corrupted, prompt #1 or #5 echoes content from a previous prompt (cross-contamination). This was the historical MKL softcap bug pattern.

## 5. Quantized KV Cache Dispatch Check

## 4. Quantized KV Cache Dispatch Check
Verify which quants route through ONEDNN vs TILE by running test-backend-ops with specific types or checking log for FA-DISP lines (if GGML_SYCL_FA_DEBUG=1 is still wired).

**Expected dispatch:**
- F16 KV → ONEDNN (native, any length)
- Q4_0-Q8_0, F32 KV + K>=1024 → ONEDNN (dequant+SDPA)
- Q4_0-Q8_0, F32 KV + K<1024 → TILE (too short for SDPA)
- BF16, IQ types → TILE (not supported by dequant)

## Why This Reporting Structure
The multi-turn Gemma test specifically catches the `stream->wait_and_throw()` bug (PR #25741) — the one that caused all the corruption in the last round. Always run it before pushing. The full benchmark catches performance regressions. test-backend-ops catches correctness regressions. These three together cover the failure modes that stalled the PR previously.
