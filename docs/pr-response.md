@ORippler The heartbeat doesn't make `ggml_backend_synchronize` heavier — it uses a dedicated private SYCL queue completely independent of the inference stream. The server-level synchronize was the original approach, but we moved the heartbeat to a separate `ggml_backend_sycl_synchronize_heartbeat` function with its own queue precisely to avoid interfering with inference.

The problem: on Windows with Intel Arc GPUs, the driver aggressively evicts VRAM from secondary/"headless" GPUs after periods of GPU inactivity. When the server wakes up from idle and tries to use the model, the device buffers are gone — the driver has to page everything back from system RAM, causing multi-second hangs or crashes on memory-constrained machines.

This is reproducible on Arc Pro B70 + B50 dual-GPU setups. The Vulkan backend has the same mechanism for the same reason (AMD/NVIDIA headless cards have similar driver behavior).

On moving it into the backend: that's a reasonable architectural option. But the heartbeat needs to fire on a timer while the server is idle — the server's idle loop is the natural place to drive it. The backend just provides the mechanism (`synchronize_heartbeat`); the server decides when to call it.

@ngxson The driver behavior has been confirmed on Windows 11 with Intel Arc (Level Zero) and Vulkan backends, on both Arc Pro B70 and B50 GPUs. Happy to provide reproduction steps if anyone wants to verify.
