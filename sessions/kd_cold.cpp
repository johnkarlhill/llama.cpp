
// kd_cold.cpp: v5 matvec on 3 rotating tensor sets (L2-cold per launch) vs 1 tensor (hot)
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <windows.h>
#include <chrono>
#pragma pack(push, 1)
struct block_pq2_0_dbg { uint16_t d; uint8_t qs[32]; };
struct block_q8_1_dbg  { uint16_t ds[2]; int8_t qs[32]; };
#pragma pack(pop)
extern "C" typedef void (*run_fn)(const void*, const void*, float*, int, int, int, uintptr_t);
static uint16_t f32h(float f) { uint32_t x; memcpy(&x,&f,4);
  uint32_t s=(x>>16)&0x8000; int e=((x>>23)&0xFF)-127+15; uint32_t m=x&0x7FFFFF;
  if(e<=0) return s; if(e>=31) return s|0x7C00; return s|(e<<10)|(m>>13); }
int main() {
  HMODULE h = LoadLibraryA("ggml-sycl.dll");
  if (!h) { printf("LoadLibrary failed %lu\n", GetLastError()); return 1; }
  auto run = (run_fn) GetProcAddress(h, "ggml_debug_pq2_0_run");
  if (!run) { printf("GetProcAddress failed\n"); return 1; }
  sycl::queue q(sycl::gpu_selector_v);
  const int TN = 17408, TC = 5120, NB = TC/128, NYB = NB*4;
  const int NSET = 3;
  std::vector<block_pq2_0_dbg*> dx(NSET); std::vector<block_q8_1_dbg*> dy(NSET);
  float* dout = (float*)sycl::malloc_device(TN*sizeof(float), q);
  srand(7);
  for (int s = 0; s < NSET; s++) {
    dx[s] = (block_pq2_0_dbg*)sycl::malloc_device(TN*NB*sizeof(block_pq2_0_dbg), q);
    dy[s] = (block_q8_1_dbg*)sycl::malloc_device(NYB*sizeof(block_q8_1_dbg), q);
    std::vector<block_pq2_0_dbg> wx(TN*NB);
    for (auto &b : wx) { b.d = f32h(0.5f + (rand()%1500)/1000.0f); for (int i=0;i<32;i++) b.qs[i]=rand()&0xFF; }
    std::vector<block_q8_1_dbg> wy(NYB);
    for (auto &b : wy) { int t=0; for (int i=0;i<32;i++){ b.qs[i]=(int8_t)(rand()%256-128); t+=b.qs[i]; }
      b.ds[0]=f32h(1.0f); b.ds[1]=f32h((float)t); }
    q.memcpy(dx[s], wx.data(), wx.size()*sizeof(block_pq2_0_dbg)).wait();
    q.memcpy(dy[s], wy.data(), wy.size()*sizeof(block_q8_1_dbg)).wait();
  }
  const int WARM=10, ITERS=60;
  auto bench = [&](const char* nm, int mode) {
    for (int i=0;i<WARM;i++) { int s = mode? (i%NSET):0; run(dx[s], dy[s], dout, TC, TN, 3, (uintptr_t)&q); }
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int i=0;i<ITERS;i++) { int s = mode? (i%NSET):0; run(dx[s], dy[s], dout, TC, TN, 3, (uintptr_t)&q); }
    q.wait();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    printf("%s: %.1f GB/s (%.3f ms/iter)\n", nm, ITERS*(double)TN*NB*sizeof(block_pq2_0_dbg)/sec/1e9, sec/ITERS*1000);
  };
  bench("hot   (1 tensor reused)", 0);
  bench("cold  (3 tensors rotate)", 1);
  return 0;
}
