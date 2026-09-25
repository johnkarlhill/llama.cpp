
// gap_bench: price the device-side inter-launch gap on the decode launch pattern.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    int layers = argc > 1 ? atoi(argv[1]) : 48;
    int reps = argc > 2 ? atoi(argv[2]) : 10;
    size_t wbytes = argc > 3 ? (size_t)atoll(argv[3]) : (size_t)56*1024*1024;
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(),
                                         sycl::property::queue::enable_profiling()});
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str()); fflush(stdout);

    size_t W16 = wbytes / 16;              // # of sycl::uint4 chunks
    auto* w = sycl::malloc_device<uint32_t>(W16*4, q);
    auto* y = sycl::malloc_device<float>(4096, q);
    q.memset(w, 1, W16*16).wait();

    // band-shaped kernel: vectorized loads, one block per slice, ~memory bound
    auto matvec = [&](int l) {
        return q.parallel_for(sycl::range<1>(256), [=](sycl::item<1> it) {
            sycl::uint4 s = {0,0,0,0};
            for (size_t i = it.get_linear_id(); i < W16; i += 256) {
                sycl::uint4 v = reinterpret_cast<const sycl::uint4*>(w)[i];
                s += v;
            }
            if (it.get_linear_id() == 0) y[l % 4096] = (float)(s[0]+s[1]+s[2]+s[3]);
        });
    };
    auto ewise = [&](int l) {
        return q.single_task([=]() { y[l % 4096] = y[l % 4096] * 0.5f + 1.0f; });
    };

    for (int l = 0; l < 8; l++) { matvec(l).wait(); ewise(l).wait(); }
    q.wait();

    double ksum=0, gsum=0; long kc=0, gc=0; std::vector<double> gaps; std::vector<double> mdurs;
    for (int r = 0; r < reps; r++) {
        std::vector<sycl::event> evs;
        for (int l = 0; l < layers; l++) { evs.push_back(matvec(l)); evs.push_back(ewise(l)); }
        q.wait();
        for (size_t k = 0; k < evs.size(); k++) {
            uint64_t s = evs[k].get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e = evs[k].get_profiling_info<sycl::info::event_profiling::command_end>();
            ksum += (e-s)/1000.0; kc++;
            if (k) { double g=(s - evs[k-1].get_profiling_info<sycl::info::event_profiling::command_end>())/1000.0; gaps.push_back(g); gsum+=g; gc++; }
            if (k%2==0) mdurs.push_back((e-s)/1000.0);
        }
    }
    std::sort(gaps.begin(), gaps.end()); std::sort(mdurs.begin(), mdurs.end());
    printf("kernels=%ld mean_dur=%.1f us (matvec median %.0f us) gaps: mean=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f us\n",
        (long)kc, ksum/kc, mdurs[mdurs.size()/2], gsum/gc, gaps[gc/2], gaps[(size_t)(gc*0.9)], gaps[(size_t)(gc*0.99)], gaps[gc-1]);
    printf("PER-TOKEN PRICE if graph is %d launches: sum_gaps=%.0f us, plus sum_matvec_dur=%0.f us\n",
        layers*2, gsum/reps, (double)0);
    return 0;
}
