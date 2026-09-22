
// standalone diff test: v3 vs generic-template PQ2_0 MMVQ kernels, identical on-device inputs
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <windows.h>
#include <chrono>

// Must match ggml-common.h block_pq2_0: fp16 d + 32 bytes qs = 34 bytes
#pragma pack(push, 1)
struct block_pq2_0_dbg { uint16_t d; uint8_t qs[32]; };
struct block_q8_1_dbg  { uint16_t ds[2]; int8_t qs[32]; };
#pragma pack(pop)

extern "C" typedef void (*run_fn)(const void*, const void*, float*, int, int, int, uintptr_t);

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16; uint32_t exp = (h >> 10) & 0x1F; uint32_t man = h & 0x3FF;
    if (exp == 0) { uint32_t x = sign; memcpy((void*)0,0,0); return 0; }
    uint32_t fexp = exp - 15 + 127; uint32_t x = sign | (fexp << 23) | (man << 13);
    float f; memcpy(&f, &x, 4); return f;
}
static uint16_t fp32_to_fp16(float f) {
    // quick IEEE half conversion
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000; int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFF;
    if (exp <= 0) return sign; if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | (man >> 13);
}



// ---- ggml-side defs needed by the embedded kernel ----
#define QK_PQ2_0 128
#define QK8_1 32
#define QI_PQ2_0 (QK_PQ2_0 / 32)
#define VDR_PQ2_0_Q8_1_MMVQ 1
#define WARP_SIZE 16
struct block_pq2_0 { uint16_t d; uint8_t qs[32]; };
struct block_q8_1_h { uint16_t ds[2]; int8_t qs[32]; };
// the embedded kernel references block_q8_1 - alias it
#define block_q8_1 block_q8_1_h

static inline float dbg_warp_reduce(const sycl::nd_item<3> & it, float v, int mask) {
    // exchange via local memory (debug kernel only)
    auto grp = it.get_group();
    return v; // replaced below
}

static inline int my_dp4a(int a, int b, int c) {
    int s = c;
    for (int i = 0; i < 4; i++) {
        int8_t ai = (int8_t)((a >> (8 * i)) & 0xFF);
        int8_t bi = (int8_t)((b >> (8 * i)) & 0xFF);
        s += (int) ai * (int) bi;
    }
    return s;
}

// ---- embedded debug copy of v3 ----
// 16-lane xor-butterfly via shared memory (debug kernel only)
static inline float dbg_xchg(const sycl::nd_item<3> & it, float v, int mask, float * shm) {
    const int lane = it.get_local_id(2);
    shm[lane] = v;
    sycl::group_barrier(it.get_sub_group());
    float o = shm[lane ^ mask];
    sycl::group_barrier(it.get_sub_group());
    return o;
}

static void mul_mat_vec_pq2_0_v3dbg(const void * __restrict__ vx,
                                    const void * __restrict__ vy,
                                    float * __restrict__ dst,
                                    const int ncols, const int nrows,
                                    const sycl::nd_item<3> & item_ct1,
                                    float * shm) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int blocks_per_row = ncols / QK_PQ2_0;
    const int lane           = item_ct1.get_local_id(2);

    const block_pq2_0 * x = (const block_pq2_0 *) vx;
    const block_q8_1_h * y = (const block_q8_1_h *) vy;

    float tmp = 0.0f;
    for (int i = 0; i < blocks_per_row; ++i) {
        const block_pq2_0 * bx = &x[row * blocks_per_row + i];
        for (int b = lane; b < QK_PQ2_0 / 4; b += WARP_SIZE) {
            const int ci = b / 8;
            const int co = (b % 8) * 4;
            const block_q8_1_h * by = &y[i * (QK_PQ2_0 / QK8_1) + ci];

            const int wb = bx->qs[b];
            const int u  = *((const int *) (by->qs + co));

            const int x0 = (wb | (wb << 12)) & 0x000F000F;
            const int qx = (x0 | (x0 << 6)) & 0x03030303;

            const int t = my_dp4a(u, qx, 0) - my_dp4a(u, 0x01010101, 0);
            const float part = (float) t * ((float) bx->d * (float) (by->ds[0]));
            if (row == 0 && i < 6 && b == lane) dst[1000 + i * 32 + lane] = part;
            tmp += part;
        }
    }

    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dbg_xchg(item_ct1, tmp, mask, shm);
    }

    if (lane == 0) dst[row] = tmp;
}

static void mul_mat_vec_pq2_0_v3dbg_sycl(const void * vx, const void * vy,
                                         float * dst, const int ncols,
                                         const int nrows, sycl::queue * stream) {
    const sycl::range<3> block_nums(1, 1, nrows);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shm_acc(32, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v3dbg(vx, vy, dst, ncols, nrows, item_ct1,
                                        shm_acc.get_pointer());
            });
    });
}

int main() {
    HMODULE h = LoadLibraryA("ggml-sycl.dll");
    if (!h) { printf("LoadLibrary failed %lu\n", GetLastError()); return 1; }
    auto run = (run_fn) GetProcAddress(h, "ggml_debug_pq2_0_run");
    if (!run) { printf("GetProcAddress failed %lu\n", GetLastError()); return 1; }

    sycl::queue q(sycl::gpu_selector_v);
    printf("queue ok\n");

    const int NCOLS = 5120;                 // K
    const int NROWS = 64;                   // out rows (small = fast)
    const int BPR   = NCOLS / 128;          // 40 weight blocks per row
    const int YBLK  = BPR * 4;              // q8_1 blocks (QK8_1=32)

    // ---- TIMING BENCH: template vs v5 at FFN scale ----
    {
        const int TN = 17408, TC = 5120;
        const int TBPR = TC / 128, TYB = TBPR * 4;
        std::vector<block_pq2_0_dbg> twx(TN * TBPR);
        std::vector<block_q8_1_dbg> twy(TYB);
        for (auto & b : twx) {
            b.d = fp32_to_fp16(0.5f + (rand() % 1500) / 1000.0f);
            for (int i = 0; i < 32; i++) b.qs[i] = rand() & 0xFF;
        }
        for (auto & b : twy) {
            int s = 0;
            for (int i = 0; i < 32; i++) { b.qs[i] = (int8_t)(rand() % 256 - 128); s += b.qs[i]; }
            b.ds[0] = fp32_to_fp16(1.0f);
            b.ds[1] = fp32_to_fp16((float)s);
        }
        float * tdx = (float *) sycl::malloc_device(twx.size() * sizeof(block_pq2_0_dbg), q);
        float * tdy = (float *) sycl::malloc_device(twy.size() * sizeof(block_q8_1_dbg), q);
        float * tdo = (float *) sycl::malloc_device(TN * sizeof(float), q);
        q.memcpy(tdx, twx.data(), twx.size() * sizeof(block_pq2_0_dbg)).wait();
        q.memcpy(tdy, twy.data(), twy.size() * sizeof(block_q8_1_dbg)).wait();
        for (int w = 1; w <= 9; ++w) { if (w==6) continue;
            const char * nm = w==1 ? "template" : (w==2 ? "v4" : (w==3 ? "v5" : (w==4 ? "v5n2" : (w==5 ? "v6" : (w==7 ? "v7" : (w==8 ? "v8" : "v9"))))));
            run(tdx, tdy, tdo, TC, TN, w, (uintptr_t)&q); q.wait();   // warmup
            auto t0 = std::chrono::steady_clock::now();
            const int ITERS = 200;
            for (int it = 0; it < ITERS; ++it) {
                run(tdx, tdy, tdo, TC, TN, w, (uintptr_t)&q);
                q.wait();
            }
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
            double gbs = (double)twx.size() * sizeof(block_pq2_0_dbg) / (ms * 1e-3) / 1e9;
            printf("BENCH %s: %.3f ms  (%.1f GB/s weight BW)\n", nm, ms, gbs);
        }
        sycl::free(tdx, q); sycl::free(tdy, q); sycl::free(tdo, q);
    }

    for (int phase = 0; phase < 3; ++phase) {
    const int which_v = (phase == 2) ? 5 : 0;
    printf("=== phase %d (v%s vs tpl) ===\n", phase, phase==2?"4":"3");
    // DETERMINISTIC test data: all codes=2 (w=+1*d), d=1; acts=+7, yds=1 -> every row dot =
    // 128*40 blocks... per block: 128 elems * (1 * 7 * 1) = 896; x 40 blocks = 35840.
    std::vector<block_pq2_0_dbg> wx(NROWS * BPR);
    for (auto & b : wx) {
        b.d = fp32_to_fp16(1.0f);
        for (int i = 0; i < 32; i++) b.qs[i] = 0xAA;   // all codes = 2 (10b)
    }
    std::vector<block_q8_1_dbg> wy(YBLK);
    srand(1234 + phase);
    if (phase == 0) {
        for (auto & b : wy) {
            b.ds[0] = fp32_to_fp16(1.0f);
            b.ds[1] = fp32_to_fp16(224.0f);
            for (int i = 0; i < 32; i++) b.qs[i] = 7;
        }
    } else {
        for (auto & b : wy) {
            int s = 0;
            for (int i = 0; i < 32; i++) { b.qs[i] = (int8_t)(rand() % 256 - 128); s += b.qs[i]; }
            b.ds[0] = fp32_to_fp16(1.0f);
            b.ds[1] = fp32_to_fp16((float)s);
        }
        for (auto & b : wx) {
            b.d = fp32_to_fp16(0.5f + (rand() % 1500) / 1000.0f);
            for (int i = 0; i < 32; i++) b.qs[i] = rand() & 0xFF;
        }
    }

    float * dx = (float *) sycl::malloc_device(wx.size() * sizeof(block_pq2_0_dbg), q);
    float * dy = (float *) sycl::malloc_device(wy.size() * sizeof(block_q8_1_dbg), q);
    float * d1 = (float *) sycl::malloc_device(NROWS * sizeof(float) + 64 * 1024, q);
    float * d2 = (float *) sycl::malloc_device(NROWS * sizeof(float) + 64 * 1024, q);
    q.memcpy(dx, wx.data(), wx.size() * sizeof(block_pq2_0_dbg)).wait();
    q.memcpy(dy, wy.data(), wy.size() * sizeof(block_q8_1_dbg)).wait();
    q.memset(d1, 0, NROWS * sizeof(float) + 64 * 1024).wait();
    q.memset(d2, 0, NROWS * sizeof(float) + 64 * 1024).wait();

    run(dx, dy, d1, NCOLS, NROWS, which_v, (uintptr_t)&q);  // variant
    q.wait();
    run(dx, dy, d2, NCOLS, NROWS, 1, (uintptr_t)&q);   // template
    q.wait();

    std::vector<float> v1(NROWS), v2(NROWS);
    q.memcpy(v1.data(), d1, NROWS * sizeof(float)).wait();
    q.memcpy(v2.data(), d2, NROWS * sizeof(float)).wait();

    // host reference for row 0
    double ref = 0.0;
    for (int i = 0; i < BPR; i++) {
        const block_pq2_0_dbg & b = wx[i];
        float d = fp16_to_fp32(b.d);
        for (int c = 0; c < 4; c++) {
            const block_q8_1_dbg & yb = wy[i*4 + c];
            float yd = fp16_to_fp32(yb.ds[0]);
            // vec_dot_pq2_0_q8_1: chunk c covers elements 32c..32c+31
            float sum = 0;
            for (int e = 0; e < 32; e++) {
                int ge = 32*c + e;              // global element in block
                int byi = ge / 4;               // weight byte
                int bit = (ge % 4) * 2;         // 2-bit code position
                int code = (b.qs[byi] >> bit) & 3;
                float w = (float)(code - 1) * d;
                sum += w * (float)yb.qs[e] * yd;
            }
            ref += sum;
        }
    }
    printf("host ref row0 = %.3f\n", ref);
    {
        // host per-block part (byte 0 only, lane 0's contribution) for blocks 0..5
        for (int i = 0; i < 6 && i < BPR; i++) {
            const block_pq2_0_dbg & b = wx[i];
            const block_q8_1_dbg & yb = wy[i * 4 + 0];   // ci=0 chunk
            float d = fp16_to_fp32(b.d);
            int t = 0;
            for (int e = 0; e < 4; e++) {
                int code = (b.qs[0] >> (2 * e)) & 3;
                t += (code - 1) * (int) yb.qs[e] * (int) fp16_to_fp32(yb.ds[0]);
            }
            printf("host block %d part0 = %.2f (d=%.4f)\n", i, (float) t * d, d);
        }
    }
    {
        std::vector<block_pq2_0_dbg> chk(2);
        q.memcpy(chk.data(), dx, 2*sizeof(block_pq2_0_dbg)).wait();
        {
        std::vector<float> parts(192);
        q.memcpy(parts.data(), d1 + 1000, 192 * sizeof(float)).wait();
        printf("v3dbg row0 parts: ");
        for (int i = 0; i < 6; i++) printf("[%d]=%.2f ", i, parts[i * 32 + 0]);
        printf("\n");
    }
    printf("dev wx0 d=%04x qs0=%02x qs1=%02x | host d=%04x %02x %02x\n",
               chk[0].d, chk[0].qs[0], chk[0].qs[1], wx[0].d, wx[0].qs[0], wx[0].qs[1]);
        std::vector<block_q8_1_dbg> chk2(2);
        q.memcpy(chk2.data(), dy, 2*sizeof(block_q8_1_dbg)).wait();
        printf("dev wy0 qs0=%d ds=%04x %04x | host qs0=%d ds=%04x %04x\n",
               (int)chk2[0].qs[0], chk2[0].ds[0], chk2[0].ds[1],
               (int)wy[0].qs[0], wy[0].ds[0], wy[0].ds[1]);
    }
    int bad = 0;
    for (int r = 0; r < NROWS; r++) {
        float diff = fabsf(v1[r] - v2[r]);
        float rel = diff / (fabsf(v2[r]) + 1e-6f);
        if (r < 8 || bad < 8) {
            if (diff > 1e-2f * (fabsf(v2[r]) + 1.0f)) {
                if (bad < 8) printf("row %d: v3=%12.4f tpl=%12.4f diff=%.4f rel=%.4f\n", r, v1[r], v2[r], diff, rel);
                bad++;
            } else if (r < 4) {
                printf("row %d: v3=%12.4f tpl=%12.4f  (match)\n", r, v1[r], v2[r]);
            }
        }
    }
    printf("MISMATCHES: %d / %d rows\n", bad, NROWS);

    sycl::free(dx, q); sycl::free(dy, q); sycl::free(d1, q); sycl::free(d2, q);
    }
    printf("ALL PHASES DONE\n");
    return 0;
}
