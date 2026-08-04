# KV Cache Pipeline, Simplified (q8_0 example)

Plain-language reference for how a quantized KV cache flows through llama.cpp on SYCL.

## The one-sentence version

K/V is computed in F16, quantized ONCE into the cache, and from then on every read dequants it back, either in bulk (prefill, for the XMX libraries) or in-register (decode, inside the kernel).

## The lifecycle of one token's K/V

| Step | What happens | When |
|---|---|---|
| 1. Compute | Model produces K/V in F16 (transient) | every prefill and decode step |
| 2. Store | F16 quantized into the cache (`ggml_set_rows`, type-converting copy) | ONCE, when the token first enters the cache |
| 3. Read, prefill | q8_0 span bulk-dequantized to dense F16 scratch, fed to oneDNN SDPA or MKL GEMM, scratch discarded | every prefill that spans this token |
| 4. Read, decode | VEC/TILE kernel reads q8_0 blocks directly from cache, dequants in-register per element | every single decode step |

A token that lives to position 100K gets dequantized roughly 100K times, quantized once.

## Mental model corrections

1. **F16 is just one cache type.** `f16`, `q8_0`, `q4_0` are all storage formats. F16 is NOT "the pipeline format".

2. **There is no "f16 pipeline".** The dense F16 materialization exists only because oneDNN SDPA and MKL GEMM are vendor libraries that require dense F16 input. It is a compatibility shim, not a pipeline requirement. Without those two paths, nothing in llama.cpp needs F16 storage: KV cache dequants in-kernel, weights dequant in-kernel (DMMV/MMQ), activations are the only F16 and they are transient.

3. **No re-quant in the normal path.** The only quant->dequant->re-quant in the codebase is `build_rope_shift` (K-shift during context overflow). Rare, deliberate, not steady-state.

4. **SDPA/GEMM = prefill accelerator, nothing more.** Both XMX paths require Q >= 32 (MKL also K >= 1024). Decode (Q=1) ALWAYS falls to VEC/TILE. The whole HYBRID/MKL machinery exists to make prefill fast.

## Why the TILE/VEC decode gate cares about quantized KV

Decode reads the whole cache every token. The cache holds quantized bytes, always. The two kernels dequant the same q8_0 blocks the same way; the only difference is work-group structure:

- VEC: ~32-64 work-groups total (one per head), each sweeps the entire cache
- TILE: ~61,000 work-groups, each reads a chunk of the cache

Same data, same dequant, totally different occupancy. On BMG at 64K-118K, TILE beats VEC by 50-68% for quantized decode. That is the entire question the gate answers.

## The holy grail (Task #31)

Fused-dequant FMHA: dequant each K/V tile in-register and feed DPAS directly, never writing F16 to HBM. One kernel, XMX-accelerated, for both prefill and decode. Q8_0 is the ideal first target because it is int8 + scale, and XMX DPAS has native int8 support (feed raw int8 to DPAS, apply scale on the accumulator).

- ESIMD `xmx::dpas` is the confirmed-viable route on BMG (joint_matrix is dead)
- Honest estimate: 10-25% at long context for prefill; decode side could be worth more
- Design pitfalls: padded seq-view KV stride, head_dim 40-512 coverage, GQA head broadcast, `sizeof(sycl::half)==2`

## Current state (2026-08-04)

- VEC: fused dequant, SIMD only, forced for quantized decode Q<=2 (gate in fattn.cpp)
- TILE: fused dequant, SIMD only, wins on BMG per A/B sweep (2026-08-04): +7.6% at 4K to +85.3% at 118K, MTP-on +14.1% at 118K, no crossover
- HYBRID/MKL/ONEDNN: XMX, dense F16 shim, prefill only
- Decode never touches the XMX paths
- Full results: see vec-tile-decode-investigation.md (RESULTS section)
