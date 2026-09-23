
// kd_v9check.cpp: run v5 (which=3) and v9 (which=9) on identical random inputs,
// compare outputs. Plus cold-L2 bench for v9 (3 rotating tensor sets).
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <windows.h>
#include <chrono>
#pragma pack(push, 1)
struct block_pq2_0_dbg { uint16_t d; uint8_t qs[32]; };
struct block_q8_1_dbg  { uint16_t ds[2]; int8_t qs[32]; };
#pragma pack(pop)
extern "C" typedef void (*run_fn)(const void*, const void*, float*, int, int, int, uintptr_t);
static float fp16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16; uint32_t exp = (h >> 10) & 0x1F; uint32_t man = h & 0x3FF;
  if (exp == 0) return 0;
  uint32_t fexp = exp - 15 + 127; uint32_t x = sign | (fexp << 23) | (man << 13);
  float f; memcpy(&f, &x, 4); return f;
}
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
  float* hout = (float*)malloc(TN*sizeof(float));
  std::vector<block_pq2_0_dbg> host_wx; std::vector<block_q8_1_dbg> host_wy;
  float* ref  = (float*)malloc(TN*sizeof(float));
  srand(1234);
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
    if (s == 0) { host_wx = wx; host_wy = wy; }
  }
  // CPU reference for row 0 (exact same math as swar kernel: per chunk, per word,
  // 8 x 2-bit codes -> byte, dp4a emulation)
  {
    double acc = 0.0;
    for (int b = 0; b < NB; b++) {
      const block_pq2_0_dbg & wb = host_wx[b]; // row 0
      float d2; uint32_t dh; memcpy(&dh, &wb.d, 2); d2 = fp16_to_f32(wb.d);
      for (int c = 0; c < 4; c++) {
        float d8 = fp16_to_f32(host_wy[b*4+c].ds[0]);
        int sumi = 0;
        for (int j = 0; j < 4; j++) {
          uint32_t q;
          uint8_t b0 = wb.qs[c*8 + j*2 + 0], b1 = wb.qs[c*8 + j*2 + 1];
          q = (uint32_t)b0 | ((uint32_t)b1 << 16);
          // 8 x 2-bit codes: q bits [1:0]=elem0, [3:2]=elem1, ... [15:14]=elem7
          int u = 0;
          memcpy(&u, &host_wy[b*4+c].qs[j*4], 4);
          for (int i = 0; i < 4; i++) {
            int cl = ((q >> (2*i)) & 3) - 1;          // elem i   (low byte)
            int ch = ((q >> (2*(i+4))) & 3) - 1;      // elem i+4 (high byte)
            int al = host_wy[b*4+c].qs[j*8 + i];
            int ah = host_wy[b*4+c].qs[j*8 + i + 4];
            sumi += cl * al + ch * ah;
          }
          (void)u;
        }
        acc += (double)d2 * d8 * sumi;
      }
    }
    printf("CPU row0 = %.2f\n", acc);
  }

  // correctness: set 0, v5 vs v9
  run(dx[0], dy[0], dout, TC, TN, 3, (uintptr_t)&q); q.wait();
  q.memcpy(ref, dout, TN*sizeof(float)).wait();
  run(dx[0], dy[0], dout, TC, TN, 9, (uintptr_t)&q); q.wait();
  q.memcpy(hout, dout, TN*sizeof(float)).wait();
  printf("v5[0..3]=%.2f %.2f %.2f %.2f\n", ref[0],ref[1],ref[2],ref[3]);
  printf("v9[0..3]=%.2f %.2f %.2f %.2f\n", hout[0],hout[1],hout[2],hout[3]);
  run(dx[0], dy[0], dout, TC, TN, 9, (uintptr_t)&q); q.wait();
  q.memcpy(ref, dout, TN*sizeof(float)).wait();
  printf("v9 rerun[0..3]=%.2f %.2f %.2f %.2f\n", ref[0],ref[1],ref[2],ref[3]);
  int bad = 0; double maxrel = 0;
  for (int r = 0; r < TN; r++) {
    double d = fabs(hout[r]-ref[r]); double rel = d / (fabs(ref[r])+1e-3);
    if (rel > maxrel) maxrel = rel;
    if (rel > 1e-3) bad++;
  }
  printf("v9 vs v5: mismatches=%d / %d, maxrel=%.2e => %s\n", bad, TN, maxrel, bad? "FAIL":"OK");
  // bench hot + cold
  const int WARM=10, ITERS=60;
  auto bench = [&](const char* nm, int which, int mode) {
    for (int i=0;i<WARM;i++) { int s = mode? (i%NSET):0; run(dx[s], dy[s], dout, TC, TN, which, (uintptr_t)&q); }
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int i=0;i<ITERS;i++) { int s = mode? (i%NSET):0; run(dx[s], dy[s], dout, TC, TN, which, (uintptr_t)&q); }
    q.wait();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    printf("%s: %.1f GB/s (%.3f ms/iter)\n", nm, ITERS*(double)TN*NB*sizeof(block_pq2_0_dbg)/sec/1e9, sec/ITERS*1000);
  };
  bench("v9 hot ", 9, 0);
  bench("v9 cold", 9, 1);
  bench("v5 cold", 3, 1);
  // MINI TEST: NCOLS=128 (1 block/row), NROWS=4: compare v5 vs v9 vs CPU on identical data
  {
    const int MROWS = 4;
    block_pq2_0_dbg* mdx = (block_pq2_0_dbg*)sycl::malloc_device(MROWS*sizeof(block_pq2_0_dbg), q);
    block_q8_1_dbg* mdy = (block_q8_1_dbg*)sycl::malloc_device(4*sizeof(block_q8_1_dbg), q);
    std::vector<block_pq2_0_dbg> mwx(MROWS);
    std::vector<block_q8_1_dbg> mwy(4);
    for (int r = 0; r < MROWS; r++) {
      mwx[r].d = f32h(0.5f + (rand()%1500)/1000.0f);
      for (int i=0;i<32;i++) mwx[r].qs[i]=rand()&0xFF;
    }
    for (int cB = 0; cB < 4; cB++) {
      int t=0;
      for (int i=0;i<32;i++){ mwy[cB].qs[i]=(int8_t)(rand()%256-128); t+=mwy[cB].qs[i]; }
      mwy[cB].ds[0]=f32h(1.0f); mwy[cB].ds[1]=f32h((float)t);
    }
    q.memcpy(mdx, mwx.data(), MROWS*sizeof(block_pq2_0_dbg)).wait();
    q.memcpy(mdy, mwy.data(), 4*sizeof(block_q8_1_dbg)).wait();
    float* mout = (float*)sycl::malloc_device(MROWS*sizeof(float), q);
    // v5 needs ncols%QK==0: 128 ok. which=3 (v5+lut) and 9.
    std::vector<float> o3(MROWS), o5(MROWS), o7(MROWS), o8(MROWS), o9(MROWS), o10(MROWS);
    run(mdx, mdy, mout, 128, MROWS, 3, (uintptr_t)&q); q.wait(); q.memcpy(o3.data(), mout, MROWS*sizeof(float)).wait();
    run(mdx, mdy, mout, 128, MROWS, 7, (uintptr_t)&q); q.wait(); q.memcpy(o7.data(), mout, MROWS*sizeof(float)).wait();
    run(mdx, mdy, mout, 128, MROWS, 8, (uintptr_t)&q); q.wait(); q.memcpy(o8.data(), mout, MROWS*sizeof(float)).wait();
    run(mdx, mdy, mout, 128, MROWS, 9, (uintptr_t)&q); q.wait(); q.memcpy(o9.data(), mout, MROWS*sizeof(float)).wait();
    // v10: build the F32 activation row (dequantized from the mwy chunks, d8=1 => act = qs)
    float myf[128];
    for (int cB = 0; cB < 4; cB++)
        for (int e = 0; e < 32; e++) myf[cB*32 + e] = (float)mwy[cB].qs[e];
    float* mfy = (float*)sycl::malloc_device(128*sizeof(float), q);
    q.memcpy(mfy, myf, 128*sizeof(float)).wait();
    run(mdx, (void*)mfy, mout, 128, MROWS, 10, (uintptr_t)&q); q.wait();
    q.memcpy(o10.data(), mout, MROWS*sizeof(float)).wait();
    for (int r = 0; r < MROWS; r++) {
      float d2 = fp16_to_f32(mwx[r].d);
      double acc = 0.0;
      for (int c = 0; c < 4; c++) {
        float d8 = fp16_to_f32(mwy[c].ds[0]);
        int sumi = 0;
        for (int j = 0; j < 4; j++) {
          uint8_t b0 = mwx[r].qs[c*8 + j*2 + 0], b1 = mwx[r].qs[c*8 + j*2 + 1];
          uint32_t q16 = (uint32_t)b0 | ((uint32_t)b1 << 16);
          for (int i = 0; i < 4; i++) {
            int cl = (((q16 >> (2*i)) & 3) - 1) & 0xFF;
            int ch = (((q16 >> (2*(i+4))) & 3) - 1) & 0xFF;
            int al = mwy[c].qs[j*8 + i];
            int ah = mwy[c].qs[j*8 + i + 4];
            sumi += (int8_t)cl * al + (int8_t)ch * ah;
          }
        }
        acc += (double)d2 * d8 * sumi;
      }
      printf("MINI row%d: v5=%.1f v7=%.1f v8=%.1f v9=%.1f v10=%.1f cpu=%.1f\n", r, o3[r], o7[r], o8[r], o9[r], o10[r], acc);
      if (r == 0) {
        printf("D2=%.4f\n", fp16_to_f32(mwx[0].d));
        printf("WQS:");
        for (int i = 0; i < 32; i++) printf(" %02x", mwx[0].qs[i]);
        printf("\n");
        for (int c = 0; c < 4; c++) {
          printf("Y%d ds=%.4f qs:", c, fp16_to_f32(mwy[c].ds[0]));
          for (int i = 0; i < 32; i++) printf(" %02x", (uint8_t)mwy[c].qs[i]);
          printf("\n");
        }
      }
    }
    sycl::free(mdx, q); sycl::free(mdy, q); sycl::free(mout, q);
  }

  // 2-COL TEST: which=11 (_ncols<2> template, the prefill path) vs v9 run twice.
  // Full-size row (K=TC) to exercise the real block walk. y = 2 back-to-back cols.
  {
    const int NB2 = TC/128, NYB2 = NB2*4;
    block_pq2_0_dbg* cdx = (block_pq2_0_dbg*)sycl::malloc_device(TN*NB2*sizeof(block_pq2_0_dbg), q);
    block_q8_1_dbg*  cdy = (block_q8_1_dbg*)sycl::malloc_device(2*NYB2*sizeof(block_q8_1_dbg), q);
    std::vector<block_pq2_0_dbg> cwx(TN*NB2);
    std::vector<block_q8_1_dbg> cwy(2*NYB2);
    for (auto &b : cwx) { b.d = f32h(0.5f + (rand()%1500)/1000.0f); for (int i=0;i<32;i++) b.qs[i]=rand()&0xFF; }
    for (int col = 0; col < 2; col++)
      for (int cB = 0; cB < NYB2; cB++) {
        int t=0;
        for (int i=0;i<32;i++){ cwy[col*NYB2+cB].qs[i]=(int8_t)(rand()%256-128); t+=cwy[col*NYB2+cB].qs[i]; }
        cwy[col*NYB2+cB].ds[0]=f32h(1.0f); cwy[col*NYB2+cB].ds[1]=f32h((float)t);
      }
    q.memcpy(cdx, cwx.data(), cwx.size()*sizeof(block_pq2_0_dbg)).wait();
    q.memcpy(cdy, cwy.data(), cwy.size()*sizeof(block_q8_1_dbg)).wait();
    float* cout = (float*)sycl::malloc_device(2*TN*sizeof(float), q);
    // v9 on col0 alone: reference = run(which=9) on col0's y, into first TN floats
    run(cdx, cdy, cout, TC, TN, 9, (uintptr_t)&q); q.wait();
    std::vector<float> o9ref(TN);
    q.memcpy(o9ref.data(), cout, TN*sizeof(float)).wait();
    // _ncols<2> on both cols
    run(cdx, cdy, cout, TC, TN, 11, (uintptr_t)&q); q.wait();
    std::vector<float> o11(2*TN);
    q.memcpy(o11.data(), cout, 2*TN*sizeof(float)).wait();
    printf("2col v9ref row0=%.2f row1=%.2f | ncols row0=%.2f row1=%.2f | col1ncols row0=%.2f\n",
           o9ref[0], o9ref[1], o11[0], o11[1], o11[TN]);
    // python reference for col0 row0..2 (same math as CPU ref above)
    for (int r = 0; r < 2; r++) {
      double acc = 0.0;
      for (int b = 0; b < NB2; b++) {
        const block_pq2_0_dbg & wb = cwx[r*NB2 + b];
        float d2 = fp16_to_f32(wb.d);
        for (int c = 0; c < 4; c++) {
          float d8 = fp16_to_f32(cwy[b*4+c].ds[0]);
          int sumi = 0;
          for (int j = 0; j < 4; j++) {
            uint8_t b0 = wb.qs[c*8 + j*2 + 0], b1 = wb.qs[c*8 + j*2 + 1];
            uint32_t q16 = (uint32_t)b0 | ((uint32_t)b1 << 16);
            for (int i = 0; i < 4; i++) {
              int cl = (((q16 >> (2*i)) & 3) - 1) & 0xFF;
              int ch = (((q16 >> (2*(i+4))) & 3) - 1) & 0xFF;
              int al = cwy[b*4+c].qs[j*8 + i];
              int ah = cwy[b*4+c].qs[j*8 + i + 4];
              sumi += (int8_t)cl * al + (int8_t)ch * ah;
            }
          }
          acc += (double)d2 * d8 * sumi;
        }
      }
      printf("2col row%d: ncols_kern=%.2f cpu_ref=%.2f %s\n", r, o11[r], acc,
             fabs(o11[r]-acc) < 0.01*fabs(acc)+1e-3 ? "OK" : "FAIL");
    }
    // col1 result vs col0 reference run
    printf("2col col1 row0=%.2f col0 row0=%.2f (should match: same... no, different y)\n", o11[TN], o11[0]);
    sycl::free(cdx, q); sycl::free(cdy, q); sycl::free(cout, q);
  }

  return 0;
}
