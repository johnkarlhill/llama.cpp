#include "mmvq.hpp"
#include <cstdio>
#include <vector>

#include "ggml.h"
#include "common.hpp"
#include "element_wise.hpp"
#include "quants.hpp"
#include "vecdotq.hpp"

// Minimum weight-row count at which the Q4_K multi-column MMVQ kernel handles two output rows per
// subgroup (rows_per_sg == 2) instead of one, when ncols_dst == 2.
//
// Pairing rows lets a subgroup load each activation block once and apply it to two rows, at the cost
// of halving the number of subgroups in the launch. With only two destination columns there is too
// little work per row to hide that loss of parallelism, so pairing only pays off once there are
// enough rows to keep the device occupied. This is a measured performance crossover, not a
// correctness or hardware limit - both variants compute the same result for any nrows.
//
// Derived on Intel Arc Pro B70 with `test-backend-ops perf -o MUL_MAT` (Q4_K, ncols_dst == 2),
// sweeping nrows over 5120..6912 at ncols 17408 and 19968: one row per subgroup was up to 9% faster
// below the crossover, two rows per subgroup 8-15% faster above it, and the crossover fell inside
// (6144, 6272] for both ncols with no measurable ncols dependence. A later 32-row granularity sweep
// narrowed it to (6144, 6176], so 6272 is a conservative gate rather than the exact crossover.
// ncols_dst >= 3 amortizes the activation loads over more columns and is faster with two rows at
// every row count, so it does not consult this threshold.
static constexpr int Q4_K_MMVQ_ROW_PAIR_MIN_NROWS = 6272;

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_reorder(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                  const int ncols, const int nrows, const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const auto         bx_offset      = block_type::get_block_offset(ibx, nblocks);
        const auto         d_offset       = block_type::get_d_offset(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t* q8_1_quant_ptr = (const int8_t*)vy + iby * QK8_1;
        const sycl::half2* q8_1_ds_ptr = (const sycl::half2*)((const char*)vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// With has_fusion, `vgate` is a second weight matrix sharing vx's shape, stride and reorder
// layout: one pass computes both row dot products and the epilogue writes glu(gate, up).
template <typename reorder_vec_dot_q_sycl, int ncols_dst, bool has_fusion = false, int rows_per_sg = 1>
static void mul_mat_vec_q_reorder_ncols(const void * __restrict__ vx, const void * __restrict__ vgate,
                                        const void * __restrict__ vy, float * __restrict__ dst, const int ncols,
                                        const int nrows, const int stride_col_y_bytes, const int stride_col_dst,
                                        const ggml_glu_op glu_op, const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row0         = (workgroup_id * sg_range + sg_id) * rows_per_sg;

    // row is sub-group uniform, so this retires whole sub-groups and the collectives below
    // stay convergent
    if (row0 >= nrows) {
        return;
    }

    static_assert(rows_per_sg == 1 ||
                  reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value);

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum[ncols_dst][rows_per_sg] = {};
    // sized 1 rather than 0 when unused: zero-length arrays are not standard C++, and the
    // array is dead and eliminated in that case
    [[maybe_unused]] float partial_gate[has_fusion ? ncols_dst : 1][has_fusion ? rows_per_sg : 1] = {};
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int  iby       = i * block_type::block_to_q8_1_ratio();

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            if constexpr (rows_per_sg > 1) {
                typename reorder_vec_dot_q_sycl::weights wx[rows_per_sg];
                [[maybe_unused]] typename reorder_vec_dot_q_sycl::weights wg[rows_per_sg];
#pragma unroll
                for (int r = 0; r < rows_per_sg; ++r) {
                    const int row = sycl::min(row0 + r, nrows - 1);
                    const int ibx = row * blocks_per_row + i;
                    const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                    const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                    wx[r] = reorder_vec_dot_q_sycl::load(vx, bx_offset, d_offset, iqs);
                    if constexpr (has_fusion) {
                        wg[r] = reorder_vec_dot_q_sycl::load(vgate, bx_offset, d_offset, iqs);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                    const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                    const sycl::half2 * q8_1_ds_ptr =
                        (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));
                    const auto a = reorder_vec_dot_q_sycl::load_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);
#pragma unroll
                    for (int r = 0; r < rows_per_sg; ++r) {
                        partial_sum[j][r] += reorder_vec_dot_q_sycl::apply(wx[r], a);
                        if constexpr (has_fusion) {
                            partial_gate[j][r] += reorder_vec_dot_q_sycl::apply(wg[r], a);
                        }
                    }
                }
            } else if constexpr (reorder_vec_dot_shared_weights<reorder_vec_dot_q_sycl::gtype>::value) {
                const int ibx = row0 * blocks_per_row + i;
                const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                const auto wx = reorder_vec_dot_q_sycl::load(vx, bx_offset, d_offset, iqs);
                if constexpr (has_fusion) {
                    const auto wg = reorder_vec_dot_q_sycl::load(vgate, bx_offset, d_offset, iqs);

#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                        const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                        const sycl::half2 * q8_1_ds_ptr =
                            (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));

                        // up and gate share the activation, so load it once and apply it twice
                        const auto a = reorder_vec_dot_q_sycl::load_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                        partial_sum[j][0] += reorder_vec_dot_q_sycl::apply(wx, a);
                        partial_gate[j][0] += reorder_vec_dot_q_sycl::apply(wg, a);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                        const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                        const sycl::half2 * q8_1_ds_ptr =
                            (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));

                        partial_sum[j][0] += reorder_vec_dot_q_sycl::dot(wx, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            } else {
                const int ibx = row0 * blocks_per_row + i;
                const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                    const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                    const sycl::half2 * q8_1_ds_ptr =
                        (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));

                    partial_sum[j][0] +=
                        reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                    if constexpr (has_fusion) {
                        partial_gate[j][0] +=
                            reorder_vec_dot_q_sycl()(vgate, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < rows_per_sg; ++r) {
            float sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum[j][r], std::plus<>());

            if constexpr (has_fusion) {
                const float gate = sycl::reduce_over_group(nd_item.get_sub_group(), partial_gate[j][r], std::plus<>());

                // uniform across the launch; the launcher only instantiates SWIGLU and GEGLU
                sum *= glu_op == GGML_GLU_OP_SWIGLU ? op_silu(gate) : op_gelu(gate);
            }

            if (sg.leader() && row0 + r < nrows) {
                dst[j * stride_col_dst + row0 + r] = sum;
            }
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                          const int ncols, const int nrows, const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    assert(blocks_per_warp > 0);

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, int ncols_dst>
static void mul_mat_vec_q_ncols(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols,
        const int nrows,
        const int stride_col_y,
        const int stride_col_dst,
        const sycl::nd_item<3> & item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1)
                  + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    // partial sums: one per output column
    float tmp[ncols_dst] = {0.0f};

    const block_q_t  * x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr);
         i < blocks_per_row;
         i += blocks_per_warp) {

        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmp[j] += dpct::permute_sub_group_by_xor(
                item_ct1.get_sub_group(), tmp[j], mask);
        }
    }

    if (item_ct1.get_local_id(2) == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            dst[j * stride_col_dst + row] = tmp[j];
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_m_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_nl_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}


template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q4_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_0, block_q4_0,
                                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, block_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q4_1_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_1, block_q4_1,
                                    VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q4_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q4_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q4_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q4_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q4_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q4_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q4_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q4_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_mxfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_MXFP4, QI_MXFP4, block_mxfp4,
                                    VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_nvfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_nvfp4_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_NVFP4, QI_NVFP4, block_nvfp4,
                                    VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_nvfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_nvfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_nvfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_nvfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_nvfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_nvfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_nvfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_nvfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, block_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q5_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK5_0, QI5_0, block_q5_0,
                                    VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q5_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q5_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q5_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q5_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q5_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q5_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q5_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q5_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, block_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q5_1_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK5_1, QI5_1, block_q5_1,
                                    VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q5_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q5_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q5_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q5_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q5_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q5_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q5_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q5_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q8_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, block_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q8_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK8_0, QI8_0, block_q8_0,
                                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q1_0_q8_1_sycl(const void * vx, const void * vy,
                                       float * dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q<QK1_0, QI1_0, block_q1_0,
                              VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1>(
                    vx, vy, dst, ncols, nrows, item_ct1);
            });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q1_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK1_0, QI1_0, block_q1_0,
                                    VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q1_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q1_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q1_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q1_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q1_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q1_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q1_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q1_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q1_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q2_0_q8_1_sycl(const void * vx, const void * vy,
                                       float * dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q<QK2_0, QI2_0, block_q2_0,
                              VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                    vx, vy, dst, ncols, nrows, item_ct1);
            });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q2_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK2_0, QI2_0, block_q2_0,
                                    VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q2_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q2_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q2_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q2_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q2_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q2_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q2_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q2_0 multi-col MMVQ", ncols_dst);
    }
}

// PQ2_0: Prism 2-bit, 128-wide blocks. The mmvq kernel iterates rows in
// QK2_0=32-element steps, so QK=32 with qi=4 (four int32 words per 128-elem
// block = 16 bytes): the vec_dot receives iqs = chunk index 0..3 and indexes
// the block itself.

// v3: PQ2_0-specialized MMVQ (tg path, ncols_dst == 1). The generic template
// instantiates PQ2_0 with qi=4, so only 4 lanes cooperate on one 128-element
// block and every q8_1 load instruction scatters its 32 lanes across 8
// different 160-byte regions - 8-16x L1 transaction amplification that keeps
// decode at ~17% of B70 bandwidth. Q6_K (qi=32, all lanes on one block)
// reaches 83% with the identical launch shape, so the launch is fine and the
// work distribution is the problem.
//
// Here all 32 lanes process ONE 128-element block per iteration:
//   lane L  -> weight byte qs[L] (elements 4L..4L+3, perfectly coalesced 32B)
//              activation 4B at chunk L/8, offset (L%8)*4 (coalesced 128B)
// Decode is a 7-op SWAR spread verified over all 256 byte values; the
// w = c-1 correction becomes an extra dp4a against 0x01010101 (sum of the 4
// activation bytes) instead of ALU work in the decode. A 256-entry LUT was
// measured SLOWER (11.35 vs 14.74 t/s) - IGC materializes constexpr tables
// as global loads - so the ALU spread stays.
static void mul_mat_vec_pq2_0_q8_1_v3(const void * __restrict__ vx,
                                      const void * __restrict__ vy,
                                      float * __restrict__ dst,
                                      const int ncols, const int nrows,
                                      const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / QK_PQ2_0;
    const int lane           = item_ct1.get_local_id(2);

    const block_pq2_0 * x = (const block_pq2_0 *) vx;
    const block_q8_1  * y = (const block_q8_1  *) vy;

    // NOTE: each 128-weight block carries its OWN fp16 scale d (v1 got this
    // right; an earlier draft hoisted block 0's d across the row and produced
    // garbage generations despite correct code decoding).
    // Each lane walks weight bytes b = lane, lane+WARP_SIZE, ... covering all
    // 32 bytes for any WARP_SIZE dividing 32 (build uses WARP_SIZE=16; the
    // original lane/8 mapping silently skipped chunks 2-3 = half the dot).
    float tmp = 0.0f;
    for (int i = 0; i < blocks_per_row; ++i) {
        const block_pq2_0 * bx = &x[row * blocks_per_row + i];
        for (int b = lane; b < QK_PQ2_0 / 4; b += WARP_SIZE) {
            const int ci = b / 8;           // q8_1 chunk (32 elems) this byte feeds
            const int co = (b % 8) * 4;     // byte offset of these 4 activations
            const block_q8_1 * by = &y[i * (QK_PQ2_0 / QK8_1) + ci];

            const int wb = bx->qs[b];
            const int u  = *((const int *) (by->qs + co));

            const int x0 = (wb | (wb << 12)) & 0x000F000F;
            const int qx = (x0 | (x0 << 6)) & 0x03030303;

            const int t = dpct::dp4a(u, qx, 0) - dpct::dp4a(u, 0x01010101, 0);
            tmp += (float) t * ((float) bx->d * (float) (by->ds[0]));
        }
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane == 0) {
        dst[row] = tmp;
    }
}


// DEBUG twin of the generic-template lane loop for PQ2_0 (WARP_SIZE=16 semantics):
// records per-lane vec_dot partials for row 0 into dst[1200..] and the row total into dst[1600+row].
static void mul_mat_vec_pq2_0_dbg_template(const void * __restrict__ vx,
                                           const void * __restrict__ vy,
                                           float * __restrict__ dst,
                                           const int ncols, const int nrows,
                                           const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (row >= nrows) return;
    const int lane = item_ct1.get_local_id(2);
    const block_pq2_0 * x = (const block_pq2_0 *) vx;
    const block_q8_1  * y = (const block_q8_1  *) vy;

    const int qi = QI_PQ2_0;                      // 4
    const int vdr = VDR_PQ2_0_Q8_1_MMVQ;          // 1
    const int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;   // 4
    const int blocks_per_row = ncols / QK_PQ2_0;  // 40
    const int stride_y = QK_PQ2_0 / QK8_1;        // 4

    float tmp = 0.0f;
    for (int i = lane / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int iqs = vdr * (lane % (qi / vdr));    // chunk 0..3
        const int ibx = row * blocks_per_row + i;
        const float s = vec_dot_pq2_0_q8_1_swar(&x[ibx], &y[i * stride_y], iqs);
        if (row == 0 && i < 4) dst[1200 + i * 16 + lane] = s;
        tmp += s;
    }
    if (row < 32) dst[1600 + row] = tmp;
}




// v6: PQ2_0 MMVQ, 2 rows per warp (halves redundant q8_1 L2 traffic).
// Template+v5 both re-read the full y vector once per output row: 111MB of L2 traffic
// per 17408x5120 matmul vs 25MB of weights. v6: lanes 0-15 stream row A's k-blocks,
// lanes 16-31 row B's; paired lanes (L, L+16) load identical y chunks -> L1 hit.
// Each lane computes the complete 4-chunk dot for one k-block of its row; two
// 16-wide sub-group reductions finish each row.
template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, int ncols_dst>
static __dpct_inline__ void mul_mat_vec_pq2_0_v6(
        const void * __restrict__ vx, const void * __restrict__ vy,
        float * __restrict__ dst, const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        const sycl::nd_item<3> & item_ct1) {

    const int lane  = item_ct1.get_local_id(2);
    const int warp  = item_ct1.get_local_id(1);
    const int row0  = (item_ct1.get_group(2) * item_ct1.get_local_range(1) + warp) * 2;
    const int sgl   = item_ct1.get_sub_group().get_group_linear_id();

    const block_q_t  * x = (const block_q_t *)  vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row = ncols / qk;
    const int sub    = lane / 16;                       // 0 = row A, 1 = row B
    const int slane  = lane % 16;                       // lane within row's 16-lane group

    const int row   = row0 + sub;
    float tmp[ncols_dst] = {0.0f};

    const int padded = (blocks_per_row + 15) / 16 * 16;
    for (int i = 0; i < padded; i += 16) {
        const int b    = i + slane;
        const bool ok  = (b < blocks_per_row) && (row < nrows);
        const int ibx  = row * blocks_per_row + (ok ? b : 0);
        const int iby0 = (ok ? b : 0) * (qk / QK8_1);

        #pragma unroll
        for (int c = 0; c < qk / QK8_1; ++c) {
            #pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmp[j] += ok ? vec_dot_q_sycl(&x[ibx], &y[j * stride_col_y + iby0], c) : 0.0f;
            }
        }
    }

    // DEBUG: lane 16 writes PRE-reduce partial (before the butterfly), so
    // a nonzero odd row here proves compute works and the reduce is broken
    // DEBUG: lane 16 writes PRE-reduce partial (before the butterfly), so
    // a nonzero odd row here proves compute works and the reduce is broken
    if (lane == 16 && row < nrows) {
        dst[row] = 42.0f + tmp[0];  // marker: proves lane16 executed
    }

    // reduce across the 16 lanes of each half
    #pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        #pragma unroll
        for (int mask = 8; mask > 0; mask >>= 1) {
            tmp[j] += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp[j], mask, 16);
        }
    }

    if (slane == 0 && row < nrows) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            dst[j * stride_col_dst + row] = tmp[j];
        }
    }
}

static void mul_mat_vec_pq2_0_q8_1_sycl_v6(const void * vx, const void * vy,
                                        float * dst, const int ncols,
                                        const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const int rows_per_warp = 2;
    const int total_warps = (nrows + rows_per_warp - 1) / rows_per_warp;
    const int warps_per_grp = 8;
    const int num_groups = (total_warps + warps_per_grp - 1) / warps_per_grp;
    const sycl::range<3> block_nums(1, 1, num_groups);
    const sycl::range<3> block_dims(1, warps_per_grp, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v6<QK_PQ2_0, QI_PQ2_0, block_pq2_0,
                                     VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1_swar, 1>(
                    vx, vy, dst, ncols, nrows, 0, 0, item_ct1);
            });
    });
}


// v5: PQ2_0 MMVQ k-parallel kernel (tg path, ncols_dst == 1).
// Template redundancy analysis: qi=4, vdr=1 -> 32 lanes compute only 4 unique chunks
// per x-block (8x redundant dp4a), warp covers 8 blocks/iter with strided 36B loads.
// v5: each lane owns a DIFFERENT x-block (32 blocks in flight), computes all 4 chunks
// itself (8x dp4a/lane, zero redundancy), loads are 32x36B = 1152B contiguous per
// warp-iteration. Sub-group reduce sums k-blocks. 4x k-parallelism vs template.
template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, int ncols_dst>
static __dpct_inline__ void mul_mat_vec_pq2_0_v5(
        const void * __restrict__ vx, const void * __restrict__ vy,
        float * __restrict__ dst, const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        const sycl::nd_item<3> & item_ct1) {

    const int lane  = item_ct1.get_local_id(2);
    const int warp  = item_ct1.get_local_id(1);                       // y-dim = warp index
    const int row   = item_ct1.get_group(2) * item_ct1.get_local_range(1) + warp;

    if (row >= nrows) return;

    const block_q_t  * x = (const block_q_t *)  vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row = ncols / qk;             // k-dim blocks
    float tmp[ncols_dst] = {0.0f};

    // lane handles k-blocks i*32+lane, stride 32 (zero redundancy, 32x36B contiguous loads)
    // full padded trip count (no break): keeps loads pipelined across iterations
    const int padded = (blocks_per_row + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    #pragma unroll 4
    for (int i = 0; i < padded; i += WARP_SIZE) {
        const int b    = i + lane;
        const bool ok  = b < blocks_per_row;
        const int ibx  = row * blocks_per_row + (ok ? b : 0);
        const int iby0 = (ok ? b : 0) * (qk / QK8_1);

        #pragma unroll
        for (int c = 0; c < qk / QK8_1; ++c) {         // 4 chunks
            #pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                float dv = ok ? vec_dot_q_sycl(&x[ibx], &y[j * stride_col_y + iby0], c) : 0.0f;
                tmp[j] += dv;
            }
        }
    }

    // sum across lanes = sum across k-blocks
    #pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        #pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmp[j] += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp[j], mask);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            dst[j * stride_col_dst + row] = tmp[j];
        }
    }
}

static void mul_mat_vec_pq2_0_q8_1_sycl_v5(const void * vx, const void * vy,
                                        float * dst, const int ncols,
                                        const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const sycl::range<3> block_nums(1, 1, nrows);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v5<QK_PQ2_0, QI_PQ2_0, block_pq2_0,
                                     VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1_swar, 1>(
                    vx, vy, dst, ncols, nrows, 0, 0, item_ct1);
            });
    });
}

// v7: v5 body (1 row/warp, full-warp reduce, proven) launched with 8 warps
// per group for occupancy. v6's 2-rows-per-warp lane split never executes
// lanes 16-31 on BMG (root cause TBD), so rows-per-warp split is abandoned.
static void mul_mat_vec_pq2_0_q8_1_sycl_v7(const void * vx, const void * vy,
                                        float * dst, const int ncols,
                                        const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const int warps_per_grp = 8;
    const int num_groups = (nrows + warps_per_grp - 1) / warps_per_grp;
    const sycl::range<3> block_nums(1, 1, num_groups);
    const sycl::range<3> block_dims(1, warps_per_grp, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v5<QK_PQ2_0, QI_PQ2_0, block_pq2_0,
                                     VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1_lut, 1>(
                    vx, vy, dst, ncols, nrows, 0, 0, item_ct1);
            });
    });
}

// v5i: v5 body with ILP-2 k-block processing (2 blocks/lane/iter, stride 64).
// Independent LUT/weight loads per pair hide global latency; reduce unchanged.
// 1 row/warp, ncols_dst == 1 only.
static __dpct_inline__ void mul_mat_vec_pq2_0_v5i(
        const void * __restrict__ vx, const void * __restrict__ vy,
        float * __restrict__ dst, const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        const sycl::nd_item<3> & item_ct1) {

    const int lane  = item_ct1.get_local_id(2);
    const int warp  = item_ct1.get_local_id(1);
    const int row   = item_ct1.get_group(2) * item_ct1.get_local_range(1) + warp;

    if (row >= nrows) return;

    const block_pq2_0  * x = (const block_pq2_0 *)  vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row = ncols / QK_PQ2_0;
    float tmp[1] = {0.0f};

    const int padded = (blocks_per_row + 2*WARP_SIZE - 1) / (2*WARP_SIZE) * (2*WARP_SIZE);
    #pragma unroll 2
    for (int i = 0; i < padded; i += 2*WARP_SIZE) {
        const bool ok0 = (i + lane) < blocks_per_row;
        const bool ok1 = (i + WARP_SIZE + lane) < blocks_per_row;
        const int b0 = ok0 ? (i + lane) : 0;
        const int b1 = ok1 ? (i + WARP_SIZE + lane) : 0;
        const int ibx0 = row * blocks_per_row + b0;
        const int ibx1 = row * blocks_per_row + b1;
        const int iby0 = b0 * (QK_PQ2_0 / QK8_1);
        const int iby1 = b1 * (QK_PQ2_0 / QK8_1);
        #pragma unroll
        for (int c = 0; c < QK_PQ2_0 / QK8_1; ++c) {
            const float dv0 = ok0 ? vec_dot_pq2_0_q8_1_lut(&x[ibx0], &y[iby0], c) : 0.0f;
            const float dv1 = ok1 ? vec_dot_pq2_0_q8_1_lut(&x[ibx1], &y[iby1], c) : 0.0f;
            tmp[0] += dv0 + dv1;
        }
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp[0] += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp[0], mask);
    }

    if (lane == 0) {
        dst[row] = tmp[0];
    }
}

// v8 launcher: v5i ILP-2 body, 8 warps/group, 1 row/warp
static void mul_mat_vec_pq2_0_q8_1_sycl_v8(const void * vx, const void * vy,
                                        float * dst, const int ncols,
                                        const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const int warps_per_grp = 8;
    const int num_groups = (nrows + warps_per_grp - 1) / warps_per_grp;
    const sycl::range<3> block_nums(1, 1, num_groups);
    const sycl::range<3> block_dims(1, warps_per_grp, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v5i(
                    vx, vy, dst, ncols, nrows, 0, 0, item_ct1);
            });
    });
}

// v9: memory-level-parallelism body. 1 row/warp, but each lane owns FOUR k-blocks
// per iteration (136B contiguous weight stream per lane, 16KB per warp) so 8+ loads
// are in flight per lane - hides cold-DRAM latency like Q6_K's template (478 GB/s
// cold). SWAR decode (LUT costs registers). Scalar pre-sum of the lane's 4 blocks,
// then the standard 32-lane butterfly.
static __dpct_inline__ void mul_mat_vec_pq2_0_v9(
        const void * __restrict__ vx, const void * __restrict__ vy,
        float * __restrict__ dst, const int ncols, const int nrows,
        const sycl::nd_item<3> & item_ct1) {

    const int lane  = item_ct1.get_local_id(2);
    const int warp  = item_ct1.get_local_id(1);
    const int row   = item_ct1.get_group(2) * item_ct1.get_local_range(1) + warp;

    if (row >= nrows) return;

    const block_pq2_0  * x = (const block_pq2_0 *)  vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row = ncols / QK_PQ2_0;
    float tmp = 0.0f;

    // blocks grouped in fours: lane L handles blocks {4L, 4L+1, 4L+2, 4L+3} per iter
    const int group_stride = WARP_SIZE * 4;
    const int padded = (blocks_per_row + group_stride - 1) / group_stride * group_stride;
    for (int i0 = 0; i0 < padded; i0 += group_stride) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int b    = i0 + lane * 4 + k;
            const bool ok  = b < blocks_per_row;
            const int ibx  = row * blocks_per_row + (ok ? b : 0);
            const int iby0 = (ok ? b : 0) * (QK_PQ2_0 / QK8_1);
            float dv = 0.0f;
            #pragma unroll
            for (int c = 0; c < QK_PQ2_0 / QK8_1; ++c) {
                dv += ok ? vec_dot_pq2_0_q8_1_swar(&x[ibx], &y[iby0], c) : 0.0f;
            }
            tmp += dv;
        }
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane == 0) {
        dst[row] = tmp;
    }
}

// v9 launcher: 1 row/warp (v5-proven shape), 1 row per group y
static void mul_mat_vec_pq2_0_q8_1_sycl_v9(const void * vx, const void * vy,
                                        float * dst, const int ncols,
                                        const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const sycl::range<3> block_nums(1, 1, nrows);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v9(
                    vx, vy, dst, ncols, nrows, item_ct1);
            });
    });
}

// v10: v9 + FUSED quantize. Takes src1 as F32; each lane quantizes its chunk of the


// v12: batched QKV — one launch covers three weight tensors sharing one q8_1 activation.
// Rows [0, nq) read wq; rows [nq, nq+nk) read wk; rows [nq+nk, nq+nk+nv) read wv.
// Saves 2 kernel launches + 2 activation-quantize round trips per full-attention layer
// per token. Same v9 inner body (SWAR, 4 blocks/lane/iter).
static __dpct_inline__ void mul_mat_vec_pq2_0_v12(
        const void * __restrict__ vq, const void * __restrict__ vk,
        const void * __restrict__ vv, const void * __restrict__ vy,
        float * __restrict__ dq, float * __restrict__ dk, float * __restrict__ dv,
        const int ncols, const int nq, const int nk, const int nv,
        const sycl::nd_item<3> & item_ct1) {

    const int lane = item_ct1.get_local_id(2);
    const int row  = item_ct1.get_group(2) * item_ct1.get_local_range(1)
                   + item_ct1.get_local_id(1);
    const int nrows_total = nq + nk + nv;
    if (row >= nrows_total) return;

    const void * w = row < nq ? vq : (row < nq + nk ? vk : vv);
    const int local_row = row < nq ? row : (row < nq + nk ? row - nq : row - nq - nk);

    const block_pq2_0  * x = (const block_pq2_0 *)  w;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row = ncols / QK_PQ2_0;
    float tmp = 0.0f;

    const int group_stride = WARP_SIZE * 4;
    const int padded = (blocks_per_row + group_stride - 1) / group_stride * group_stride;
    for (int i0 = 0; i0 < padded; i0 += group_stride) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int b    = i0 + lane * 4 + k;
            const bool ok  = b < blocks_per_row;
            const int ibx  = local_row * blocks_per_row + (ok ? b : 0);
            const int iby0 = (ok ? b : 0) * (QK_PQ2_0 / QK8_1);
            float dv = 0.0f;
            #pragma unroll
            for (int c = 0; c < QK_PQ2_0 / QK8_1; ++c) {
                dv += ok ? vec_dot_pq2_0_q8_1_swar(&x[ibx], &y[iby0], c) : 0.0f;
            }
            tmp += dv;
        }
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane == 0) {
        float * d = row < nq ? dq : (row < nq + nk ? dk : dv);
        d[local_row] = tmp;
    }
}

static void mul_mat_vec_pq2_0_batched_sycl_v12(const void * vq, const void * vk, const void * vv,
                                        const void * vy, float * dq, float * dk, float * dv,
                                        const int ncols, const int nq, const int nk, const int nv,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const int nrows_total = nq + nk + nv;
    const sycl::range<3> block_nums(1, 1, nrows_total);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v12(vq, vk, vv, vy, dq, dk, dv, ncols, nq, nk, nv, item_ct1);
            });
    });
}

// v10: v9 + FUSED quantize. Takes src1 as F32; each lane quantizes its chunk of the
// activation row in registers (same math as quantize_q8_1: amax -> d = amax/127 ->
// int8 round), so no separate quantize kernel runs between matvecs. 4 k-blocks per
// lane per iteration (v9 MLP structure), SWAR weight decode.
static __dpct_inline__ void mul_mat_vec_pq2_0_v10(
        const void * __restrict__ vx, const float * __restrict__ yf,
        float * __restrict__ dst, const int ncols, const int nrows,
        const sycl::nd_item<3> & item_ct1) {

    const int lane  = item_ct1.get_local_id(2);
    const int row   = item_ct1.get_group(2) * item_ct1.get_local_range(1)
                    + item_ct1.get_local_id(1);

    if (row >= nrows) return;

    const block_pq2_0 * x = (const block_pq2_0 *) vx;

    const int blocks_per_row = ncols / QK_PQ2_0;
    float tmp = 0.0f;

    const int group_stride = WARP_SIZE * 4;
    const int padded = (blocks_per_row + group_stride - 1) / group_stride * group_stride;
    for (int i0 = 0; i0 < padded; i0 += group_stride) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int b    = i0 + lane * 4 + k;
            const bool ok  = b < blocks_per_row;
            const int ibx  = row * blocks_per_row + (ok ? b : 0);
            const float * ychunk = yf + (ok ? b : 0) * QK_PQ2_0;

            float sumf = 0.0f;
            #pragma unroll
            for (int c = 0; c < QK_PQ2_0 / QK8_1; ++c) {
                if (!ok) break;
                // ---- fused quantize of 32 activations (chunk c) ----
                const int base = c * QK8_1;
                float amax = 0.0f;
                float av[QK8_1];
                #pragma unroll
                for (int e = 0; e < QK8_1; ++e) {
                    av[e] = ychunk[base + e];
                    amax = sycl::fmax(amax, sycl::fabs(av[e]));
                }
                // warp-wide amax: lane holds 32 elems? NO - each lane owns the whole
                // chunk, so amax is already the full-chunk max (no exchange needed).
                const float d  = (amax == 0.0f) ? 1.0f : amax / 127.0f;
                const float id = (amax == 0.0f) ? 0.0f : 127.0f / amax;
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint16_t w0 = *(const uint16_t *)(x[ibx].qs + c*8 + j*2);
                    int u = 0, qx = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int cl = (((w0 >> (2*i)) & 3) - 1) & 0xFF;
                        const int ch = (((w0 >> (2*(i+4))) & 3) - 1) & 0xFF;
                        qx |= cl << (8*i);
                        u  |= (((int)sycl::round(av[8*j + i]   * id)) & 0xFF) << (8*i);
                        u  |= (((int)sycl::round(av[8*j + i+4] * id)) & 0xFF) << (8*(i+4));
                    }
                    sumi = dpct::dp4a(u, qx, sumi);
                }
                sumf += (amax == 0.0f ? 0.0f : d) * sumi;
            }
            tmp += ok ? sumf : 0.0f;
        }
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane == 0) {
        dst[row] = tmp;
    }
}

// v10 launcher: 1 row/warp
static void mul_mat_vec_pq2_0_q8_1_sycl_v10(const void * vx, const float * yf,
                                        float * dst, const int ncols,
                                        const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const sycl::range<3> block_nums(1, 1, nrows);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v10(vx, yf, dst, ncols, nrows, item_ct1);
            });
    });
}

// v11: correctness experiment - pure FP32 activation dot. Decodes 2-bit weights to
// {-1,0,1} and FMAs against the F32 activation directly. NO activation quantization
// anywhere: isolates whether q8_1 quantize noise is what flips "following" vs "Paris".
// Env-gated GGML_SYCL_PQ2_F32ACT=1. Perf-irrelevant by design (fp32 FMA, no dp4a).
static __dpct_inline__ void mul_mat_vec_pq2_0_v11(
        const void * __restrict__ vx, const float * __restrict__ yf,
        float * __restrict__ dst, const int ncols, const int nrows,
        const sycl::nd_item<3> & item_ct1) {

    const int lane  = item_ct1.get_local_id(2);
    const int row   = item_ct1.get_group(2) * item_ct1.get_local_range(1)
                    + item_ct1.get_local_id(1);

    if (row >= nrows) return;

    const block_pq2_0 * x = (const block_pq2_0 *) vx;

    const int blocks_per_row = ncols / QK_PQ2_0;
    float tmp = 0.0f;

    const int group_stride = WARP_SIZE * 4;
    const int padded = (blocks_per_row + group_stride - 1) / group_stride * group_stride;
    for (int i0 = 0; i0 < padded; i0 += group_stride) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int b    = i0 + lane * 4 + k;
            const bool ok  = b < blocks_per_row;
            const int ibx  = row * blocks_per_row + (ok ? b : 0);
            const float * ychunk = yf + (ok ? b : 0) * QK_PQ2_0;

            float sumf = 0.0f;
            if (ok) {
                const float d2 = x[ibx].d;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint16_t w0 = *(const uint16_t *)(x[ibx].qs + j*2);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float cl = (float)(((w0 >> (2*i)) & 3) - 1);
                        const float ch = (float)(((w0 >> (2*(i+4))) & 3) - 1);
                        sumf += cl * ychunk[8*j + i] + ch * ychunk[8*j + i + 4];
                    }
                }
                sumf *= d2;
            }
            tmp += sumf;
        }
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane == 0) {
        dst[row] = tmp;
    }
}

// v11 launcher: 1 row/warp (same shape as v9/v10)
static void mul_mat_vec_pq2_0_q8_1_sycl_v11(const void * vx, const float * yf,
                                        float * dst, const int ncols,
                                        const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const sycl::range<3> block_nums(1, 1, nrows);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v11(vx, yf, dst, ncols, nrows, item_ct1);
            });
    });
}

// v4: PQ2_0 MMVQ with Q6_K-like access pattern (tg path, ncols_dst == 1).
// 8 lanes per 128-elem block, 2 blocks per warp iteration:
//   lane L in group g (g = L/8), block pair (2i + g):
//     weight word w = L%8 (int32 = 4 bytes = 16 codes; 8 lanes x 4B = 32B coalesced)
//     activations: chunk c = w/2, half h = (w%2)*16 -> 16B read; 8 lanes cover the
//     block's 4 chunks = 128B coalesced.
//   Decode: per byte, the same 7-op SWAR spread as v3, one dp4a per byte.
//   Reduce: 8-lane butterflies (masks 4/2/1; mask 8 would mix the two blocks).
//   After the loop, group totals ride to lane 0 via one mask-8 exchange and are
//   written as two separate stores (no dst accumulation races).
static void mul_mat_vec_pq2_0_q8_1_v4(const void * __restrict__ vx,
                                      const void * __restrict__ vy,
                                      float * __restrict__ dst,
                                      const int ncols, const int nrows,
                                      const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / QK_PQ2_0;
    const int lane           = item_ct1.get_local_id(2);

    const block_pq2_0 * x = (const block_pq2_0 *) vx;
    const block_q8_1  * y = (const block_q8_1  *) vy;

    const int g  = lane / 8;         // which block of the pair
    const int w  = lane % 8;         // weight word within the block
    const int ci = w / 2;            // q8_1 chunk (32 elems)
    const int ch = (w % 2) * 16;     // 16B half-chunk offset

    float tmp = 0.0f;
    for (int i0 = 0; i0 < blocks_per_row; i0 += 2) {
        const int i = i0 + g;
        if (i >= blocks_per_row) {
            break;
        }
        const block_pq2_0 * bx = &x[row * blocks_per_row + i];
        const block_q8_1  * by = &y[i * (QK_PQ2_0 / QK8_1) + ci];

        const int wb4     = *((const int *) (bx->qs + w * 4));
        const float scale = (float) bx->d * (float) (by->ds[0]);

        float part = 0.0f;
        #pragma unroll
        for (int byi = 0; byi < 4; ++byi) {
            const int wb = (wb4 >> (8 * byi)) & 0xFF;
            const int u  = *((const int *) (by->qs + ch + byi * 4));

            const int x0 = (wb | (wb << 12)) & 0x000F000F;
            const int qx = (x0 | (x0 << 6)) & 0x03030303;

            const int t = dpct::dp4a(u, qx, 0) - dpct::dp4a(u, 0x01010101, 0);
            part += (float) t * scale;
        }

        // 8-lane butterfly (masks 4/2/1 stay within each block's group)
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            part += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), part, mask);
        }
        tmp += part;
    }

    // fold the two group totals: one mask-8 xor exchange merges both groups'
    // totals into every lane; group 0's lane 0 writes the row sum.
    tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, 8);

    if (w == 0) {
        dst[row] = tmp;
    }
}

static void mul_mat_vec_pq2_0_q8_1_v4_sycl(const void * vx, const void * vy,
                                           float * dst, const int ncols,
                                           const int nrows,
                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const sycl::range<3> block_nums(1, 1, nrows);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v5<QK_PQ2_0, QI_PQ2_0, block_pq2_0, VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1_lut, 1>(vx, vy, dst, ncols, nrows, 0, 0, item_ct1);
            });
    });
}

static void mul_mat_vec_pq2_0_q8_1_v3_sycl(const void * vx, const void * vy,
                                           float * dst, const int ncols,
                                           const int nrows,
                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const sycl::range<3> block_nums(1, 1, nrows);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v6<QK_PQ2_0, QI_PQ2_0, block_pq2_0, VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1_swar, 1>(vx, vy, dst, ncols, nrows, 0, 0, item_ct1);
            });
    });
}

static void mul_mat_vec_pq2_0_q8_1_sycl(const void * vx, const void * vy,
                                        float * dst, const int ncols,
                                        const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v5<QK_PQ2_0, QI_PQ2_0, block_pq2_0, VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1_lut, 1>(vx, vy, dst, ncols, nrows, 0, 0, item_ct1);
            });
    });
}

// template-path fallback (GGML_SYCL_PQ2_V6=0): upstream generic ncols kernel
static void mul_mat_vec_pq2_0_q8_1_sycl_tpl(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        dpct::queue_ptr stream) {
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q<QK_PQ2_0, QI_PQ2_0, block_pq2_0,
                              VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1_swar>(
                    vx, vy, dst, ncols, nrows, item_ct1);
            });
    });
}

template <int ncols_dst>
static void mul_mat_vec_pq2_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_PQ2_0, QI_PQ2_0, block_pq2_0,
                                    VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1_swar, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

template <int ncols_dst>
static void mul_mat_vec_pq2_0_q8_1_sycl_v6n(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const int rows_per_warp = 2;
    const int total_warps = (nrows + rows_per_warp - 1) / rows_per_warp;
    const int warps_per_grp = 8;
    const int num_groups = (total_warps + warps_per_grp - 1) / warps_per_grp;
    const sycl::range<3> block_nums(1, 1, num_groups);
    const sycl::range<3> block_dims(1, warps_per_grp, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v6<QK_PQ2_0, QI_PQ2_0, block_pq2_0,
                                     VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1_swar, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

template <int ncols_dst>
static void mul_mat_vec_pq2_0_q8_1_sycl_v5n(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("Calling pq2_0 v5n<%d> nrows=%d ncols=%d\n", ncols_dst, nrows, ncols);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_pq2_0_v5<QK_PQ2_0, QI_PQ2_0, block_pq2_0,
                                     VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1_swar, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_pq2_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    // A/B: default = v4-era template path (last known good gen). GGML_SYCL_PQ2_V6=1 opts
    // into the v6 two-rows-per-warp kernel (2.4x isolated; 2-col path NOT yet validated).
    static const bool use_v6 = getenv("GGML_SYCL_PQ2_V6") != nullptr
        && strcmp(getenv("GGML_SYCL_PQ2_V6"), "1") == 0;
    if (!use_v6) {
        switch (ncols_dst) {
            case 1: mul_mat_vec_pq2_0_q8_1_sycl_tpl(vx, vy, dst, ncols, nrows, stream); break;
            default: mul_mat_vec_pq2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        }
        return;
    }
    switch (ncols_dst) {
        case 1: mul_mat_vec_pq2_0_q8_1_sycl_v9(vx, vy, dst, ncols, nrows, stream); break;  // 48t/s push: v9 MLP body
        case 2: mul_mat_vec_pq2_0_q8_1_sycl_v6n<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_pq2_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_pq2_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_pq2_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_pq2_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_pq2_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_pq2_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for PQ2_0 multi-col MMVQ", ncols_dst);
    }
}

// PTQ1_0: Prism packed-trit, 128-wide blocks, whole block per vec_dot call
// (VDR 4 = four q8_1 chunks). Kernel iterates in 32-element steps so QK=32,
// qi=1 (one int32 word read per step from the 26-byte payload).
static void mul_mat_vec_ptq1_0_q8_1_sycl(const void * vx, const void * vy,
                                         float * dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PTQ1_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q<QK_PTQ1_0, QI_PTQ1_0, block_ptq1_0,
                              VDR_PTQ1_0_Q8_1_MMVQ, vec_dot_ptq1_0_q8_1>(
                    vx, vy, dst, ncols, nrows, item_ct1);
            });
    });
}

template <int ncols_dst>
static void mul_mat_vec_ptq1_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PTQ1_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_PTQ1_0, QI_PTQ1_0, block_ptq1_0,
                                    VDR_PTQ1_0_Q8_1_MMVQ, vec_dot_ptq1_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_ptq1_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_ptq1_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_ptq1_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_ptq1_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_ptq1_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_ptq1_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_ptq1_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_ptq1_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_ptq1_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for PTQ1_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q2_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_K, QI2_K, block_q2_K,
                                    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q2_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q2_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q2_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q2_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q2_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q2_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q2_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q2_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q2_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q2_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q2_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q3_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

template <int ncols_dst>
static void mul_mat_vec_q3_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_K, QI3_K, block_q3_K,
                                    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q3_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q3_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q3_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q3_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q3_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q3_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q3_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q3_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
}


static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q4_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_K, block_q4_K,
                                        VDR_Q4_K_Q8_1_MMVQ,
                                        vec_dot_q4_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q4_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q4_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q4_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q4_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q4_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q4_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q4_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q4_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

template <int ncols_dst, int rows_per_sg>
static void reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups * rows_per_sg);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>, ncols_dst,
                                                        /*has_fusion=*/ false, rows_per_sg>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    constexpr int rows_per_sg = ncols_dst >= 3 && ncols_dst <= 4 ? 2 : 1;
    reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<ncols_dst, rows_per_sg>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2:
            if (nrows >= Q4_K_MMVQ_ROW_PAIR_MIN_NROWS) {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            } else {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 1>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            }
            break;
        case 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q5_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI5_K, block_q5_K,
                                        VDR_Q5_K_Q8_1_MMVQ,
                                        vec_dot_q5_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q5_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q5_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q5_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q5_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q5_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q5_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q5_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q5_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q5_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);


    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q6_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI6_K, block_q6_K,
                                        VDR_Q6_K_Q8_1_MMVQ,
                                        vec_dot_q6_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q6_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q6_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q6_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q6_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q6_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q6_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q6_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q6_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
}


static void mul_mat_vec_iq2_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, block_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_xs_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, block_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_s_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, block_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, block_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, block_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, block_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_m_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, block_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_nl_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, block_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_xs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, block_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_XS/4, block_iq4_xs,
                                        1,
                                        vec_dot_iq4_xs_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_col_size,
                                const dpct::queue_ptr & stream) {
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    static int pq2_dump_count = 0;
    static int n_yb_global = 0;
    (void) n_yb_global;

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    for (int i = 0; i < src1_ncols; i++) {
        const size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        const char * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        float *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q4_1:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_0:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_1:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q8_0:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q1_0:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl\n");
                    mul_mat_vec_q1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q2_0:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl\n");
                    mul_mat_vec_q2_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_PQ2_0:
                // A reordered (SoA) tensor must never reach the AOS v9 body: silent wrong
                // math. Reordered PQ2_0 weights serve the SoA switch_ncols kernel below
                // (cols 1..8) and the fused-GLU kernel; if a reordered tensor escapes to
                // the AOS body (e.g. prefill cols>8 after a decode reordered in place),
                // fail loud, do not compute garbage.
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8 && getenv("GGML_SYCL_PQ2_NO_MMQ") == nullptr) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_pq2_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_pq2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    {
                        static const bool pq2_dump2 = getenv("GGML_SYCL_PQ2_DUMP") != nullptr
                            && getenv("GGML_SYCL_PQ2_DUMP")[0] == '1';
                        static int dump2_count = 0;
                        if (pq2_dump2 && i == 0 && dump2_count < 3) {
                            dump2_count++;
                            char fn2[128];
                            snprintf(fn2, sizeof(fn2), "C:\\llama.cpp-build-sycl\\pq2_pre_%d.bin", dump2_count);
                            FILE * f2 = fopen(fn2, "wb");
                            if (f2) {
                                int m2[8] = { (int)ne10, (int)src1_padded_col_size, (int)row_diff,
                                              (int)ne00, (int)src1_ncols, (int)src0->ne[1], 0, 0 };
                                std::vector<char> xb2(320 * sizeof(block_pq2_0));
                                stream->memcpy(xb2.data(), src0_dd_i, xb2.size()).wait();
                                size_t yb2_sz = (size_t)(src1_padded_col_size / QK8_1) * sizeof(block_q8_1) * src1_ncols;
                                std::vector<char> yb2(yb2_sz);
                                stream->memcpy(yb2.data(), src1_ddq_i, yb2_sz).wait();
                                std::vector<float> db2(std::min<size_t>(64, (size_t)row_diff) * src1_ncols);
                                stream->memcpy(db2.data(), dst_dd_i, db2.size()*4).wait();
                                fwrite(m2, 4, 8, f2);
                                fwrite(xb2.data(), 1, xb2.size(), f2);
                                fwrite(yb2.data(), 1, yb2_sz, f2);
                                fwrite(db2.data(), 4, db2.size(), f2);
                                fclose(f2);
                                printf("PQ2_PRE_%d ne10=%d pad=%d ncols=%d rowdiff=%d\n",
                                       dump2_count, ne10, (int)src1_padded_col_size, (int)src1_ncols, (int)row_diff);
                            }
                        }
                    }
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    // AOS body: unreachable for reordered weights (cols 1..8 route to
                    // switch_ncols above); a reordered tensor here means prefill-after-
                    // decode — fail loud instead of computing garbage.
                    GGML_ASSERT(!(((ggml_tensor_extra_gpu *) dst->src[0]->extra) &&
                                  ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder));
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_pq2_0_q8_1_sycl\n");
                    // A/B: v9 (pure SWAR MLP, reads pre-quantized q8_1) — v10 fused
                    // parked until its quantize math is validated (MINI: returns 0)
                    static const bool pq2_f32act = getenv("GGML_SYCL_PQ2_F32ACT") != nullptr
                        && getenv("GGML_SYCL_PQ2_F32ACT")[0] == '1';
                    if (pq2_f32act) {
                        // correctness experiment: F32 activations, no quantize noise
                        const float * src1_ddf_i_bs = src1_ddf_i + i * src1_padded_col_size * sizeof(float);
                        mul_mat_vec_pq2_0_q8_1_sycl_v11(src0_dd_i, src1_ddf_i_bs, dst_dd_i_bs,
                            ne00, row_diff, stream);
                    } else {
                        mul_mat_vec_pq2_0_q8_1_sycl_v9(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs,
                            ne00, row_diff, stream);
                    }
                }
                {
                    static const bool pq2_dump = getenv("GGML_SYCL_PQ2_DUMP") != nullptr
                        && getenv("GGML_SYCL_PQ2_DUMP")[0] == '1';
                    if (pq2_dump && i == 0 && src1_ncols == 1 && src0->ne[1] == 248320 && pq2_dump_count < 6) {
                        pq2_dump_count++;
                        char fn[128];
                        snprintf(fn, sizeof(fn), "C:\\llama.cpp-build-sycl\\pq2_dump_%d.bin", pq2_dump_count);
                        FILE * fdbg = fopen(fn, "wb");
                        if (fdbg) {
                            int meta[12];
                            meta[0] = (int) ne10;                    // k dim
                            meta[1] = (int) src1_padded_col_size;
                            meta[2] = (int) row_diff;
                            meta[3] = (int) ne00;
                            meta[4] = (int) src0->ne[1];             // nrows of weights
                            meta[5] = (int) dst->ne[0];
                            meta[6] = ggml_get_op_params_i32(dst, 1);// op param hint
                            meta[7] = (int) dst->op;
                            meta[8] = (int) (n_yb_global);
                            meta[9] = pq2_dump_count;
                            meta[10] = (int) src1->ne[0];
                            meta[11] = (int) src0->ne[0];
                            const size_t blk_sz = sizeof(block_pq2_0);
                            std::vector<float> xf(ne10);
                            stream->memcpy(xf.data(), src1_ddf_i, ne10*4).wait();
                            fwrite(xf.data(), 4, ne10, fdbg);
                            const size_t y8_sz  = sizeof(block_q8_1);
                            std::vector<char> xb(320 * blk_sz);
                            stream->memcpy(xb.data(), src0_dd_i, xb.size()).wait();
                            std::vector<char> yb((size_t)std::max(160, (int)(src1_padded_col_size / QK8_1)) * y8_sz * (src1_ncols > 1 ? src1_ncols : 1));
                            stream->memcpy(yb.data(), src1_ddq_i, yb.size()).wait();
                            std::vector<float> db(std::max(64, (int)row_diff < 64 ? (int)row_diff : 64) * (src1_ncols > 1 ? src1_ncols : 1));
                            stream->memcpy(db.data(), dst_dd_i, db.size()*4).wait();
                            fwrite(meta, 4, 12, fdbg);
                            fwrite(xb.data(), 1, xb.size(), fdbg);
                            fwrite(yb.data(), 1, yb.size(), fdbg);
                            fwrite(db.data(), 4, db.size(), fdbg);
                            fclose(fdbg);
                            printf("PQ2_DUMP_%d ne10=%d pad=%d rowdiff=%d ne00=%d op=%d hint=%d\n",
                                   pq2_dump_count, meta[0], meta[1], meta[2], meta[3], meta[7], meta[6]);
                        }
                    }
                }
                break;
            case GGML_TYPE_PTQ1_0:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_ptq1_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_ptq1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_ptq1_0_q8_1_sycl\n");
                    mul_mat_vec_ptq1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q2_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q2_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl\n");
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q3_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q4_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q6_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_XXS:
                mul_mat_vec_iq3_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_S:
                mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_XS:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_MXFP4:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_NVFP4:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            default:
                GGML_ABORT("fatal error: unsupport src0 data type %s\n", ggml_type_name(src0->type));
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}

// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
static __dpct_inline__ float vec_dot_iq2_xxs_q8_1_moe(const void * __restrict__ vbq,
                                                      const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

static __dpct_inline__ float vec_dot_iq2_xs_q8_1_moe(const void * __restrict__ vbq,
                                                     const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid, ksigns64);
}

static __dpct_inline__ float vec_dot_iq3_xxs_q8_1_moe(const void * __restrict__ vbq,
                                                      const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid, ksigns64);
}

static __dpct_inline__ float vec_dot_iq3_s_q8_1_moe(const void * __restrict__ vbq,
                                                    const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

static __dpct_inline__ float vec_dot_iq1_s_q8_1_moe(const void * __restrict__ vbq,
                                                    const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
}

// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q_moe(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    float * __restrict__ dst_base, const int32_t * __restrict__ ids_dev,
    const int ncols, const int nrows,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride,
    const sycl::nd_item<3> & item_ct1) {

    const int expert_idx = item_ct1.get_group(1);
    const int i02        = ids_dev[expert_idx];

    const char * vx = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
    const char * vy = (const char *) vy_base + (size_t) expert_idx * src1_row_stride;
    float *      dst = (float *) ((char *) dst_base + (size_t) expert_idx * dst_row_stride);

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe(
    const void * vx_base, const void * vy, const int32_t * ids_dev,
    float * dst_base, const int ncols, const int nrows, const int n_experts_used,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride,
    dpct::queue_ptr stream) {
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, (unsigned) n_experts_used, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                    vx_base, vy, dst_base, ids_dev, ncols, nrows,
                    expert_weight_stride, dst_row_stride, src1_row_stride, item);
            });
    });
}

bool ggml_sycl_mul_mat_vec_q_id(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream) {
    switch (src0_type) {
        case GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q2_0:
            launch_mul_mat_vec_q_moe<QK2_0, QI2_0, block_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_PQ2_0:
            // qk = QK_PQ2_0: block_pq2_0 spans 128 elements; 4 lanes per block,
            // each lane handles one 32-element chunk (iqs = lane%4).
            launch_mul_mat_vec_q_moe<QK_PQ2_0, QI_PQ2_0, block_pq2_0, VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_PTQ1_0:
            // qk = QK_PTQ1_0, vdr = 4: one lane per 128-element block consumes
            // the whole block + its four q8_1 chunks in a single call.
            launch_mul_mat_vec_q_moe<QK_PTQ1_0, QI_PTQ1_0, block_ptq1_0, VDR_PTQ1_0_Q8_1_MMVQ, vec_dot_ptq1_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ2_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XXS/2, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ2_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XS/2, block_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ2_S:
            launch_mul_mat_vec_q_moe<QK_K, QI2_S/2, block_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ3_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI3_XXS/2, block_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ3_S:
            launch_mul_mat_vec_q_moe<QK_K, QI3_S/2, block_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ1_S:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, block_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ1_M:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, block_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_vec_q_moe<QK4_NL, QI4_NL, block_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ4_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI4_XS/4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        default:
            return false;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_reorder(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    float * __restrict__ dst_base, const int32_t * __restrict__ ids_dev,
    const int ncols, const int nrows,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride,
    const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const int expert_idx = item_ct1.get_group(1);
    const int i02        = ids_dev[expert_idx];

    const char * vx  = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
    const char * vy  = (const char *) vy_base + (size_t) expert_idx * src1_row_stride;
    float *      dst = (float *) ((char *) dst_base + (size_t) expert_idx * dst_row_stride);

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const auto sg = item_ct1.get_sub_group();

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;

        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);

        const int           iby            = i * block_type::block_to_q8_1_ratio();
        const int8_t *      q8_1_quant_ptr = (const int8_t *) vy + iby * QK8_1;
        const sycl::half2 * q8_1_ds_ptr    = (const sycl::half2 *) ((const char *) vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);
            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());
    if (sg.leader()) {
        dst[row] = sum;
    }
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_reorder(
    const void * vx_base, const void * vy, const int32_t * ids_dev,
    float * dst_base, const int ncols, const int nrows, const int n_experts_used,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride,
    dpct::queue_ptr stream) {
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, (unsigned) n_experts_used, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>(
                    vx_base, vy, dst_base, ids_dev, ncols, nrows,
                    expert_weight_stride, dst_row_stride, src1_row_stride, item);
            });
    });
}

bool ggml_sycl_mul_mat_vec_q_id_reorder(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream) {
    switch (src0_type) {
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        default:
            return false;
    }
}

template <typename reorder_vec_dot_q_sycl, int ncols_dst, int rows_per_sg>
static void launch_mul_mat_vec_q_reorder_glu_impl(const void * vx, const void * vgate, const void * vy, float * dst,
                                             const int ncols, const int nrows, const int stride_col_y_bytes,
                                             const int stride_col_dst, const ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = WARP_SIZE;

    const int            block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups * rows_per_sg);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl, ncols_dst, /*has_fusion=*/ true,
                                                        rows_per_sg>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op,
                                 nd_item);
                         });
    });
}

template <typename reorder_vec_dot_q_sycl, int ncols_dst>
static void launch_mul_mat_vec_q_reorder_glu(const void * vx, const void * vgate, const void * vy, float * dst,
                                             const int ncols, const int nrows, const int stride_col_y_bytes,
                                             const int stride_col_dst, const ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    constexpr int rows_per_sg =
        reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value && ncols_dst >= 3 && ncols_dst <= 4
            ? 2
            : 1;
    launch_mul_mat_vec_q_reorder_glu_impl<reorder_vec_dot_q_sycl, ncols_dst, rows_per_sg>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op, stream);
}

static bool ggml_sycl_mul_mat_vec_q_glu_reorder_pq2(enum ggml_glu_op glu_op, const void * vx,
                                                     const void * vgate, const void * vy, float * dst,
                                                     int ncols, int nrows, int ncols_dst,
                                                     int stride_col_y_bytes, int stride_col_dst,
                                                     dpct::queue_ptr stream) {
    using vec_dot = reorder_vec_dot_q_sycl<GGML_TYPE_PQ2_0>;

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 1>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 2:
            // rows_per_sg=2 requires reorder_vec_dot_shared_activations<PQ2_0> (not
            // specialized); stay at 1 row/subgroup for v1.
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 2>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        default:
            return false;
    }
}

bool ggml_sycl_mul_mat_vec_q_glu_reorder(enum ggml_type src0_type, enum ggml_glu_op glu_op, const void * vx,
                                         const void * vgate, const void * vy, float * dst, int ncols, int nrows,
                                         int ncols_dst, int stride_col_y_bytes, int stride_col_dst,
                                         dpct::queue_ptr stream) {
    if (src0_type != GGML_TYPE_Q4_K && src0_type != GGML_TYPE_PQ2_0) {
        return false;
    }
    // f32-activation experiment: clean A/B requires the plain v9/v11 route everywhere
    if (getenv("GGML_SYCL_PQ2_F32ACT") != nullptr && getenv("GGML_SYCL_PQ2_F32ACT")[0] == '1') {
        return false;
    }
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        return false;
    }

    switch (src0_type) {
        case GGML_TYPE_PQ2_0:
            return ggml_sycl_mul_mat_vec_q_glu_reorder_pq2(glu_op, vx, vgate, vy, dst, ncols, nrows,
                                                           ncols_dst, stride_col_y_bytes, stride_col_dst, stream);
        default:
            break;
    }

    using vec_dot = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>;

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 1>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 2:
            if (nrows >= Q4_K_MMVQ_ROW_PAIR_MIN_NROWS) {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 2>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op, stream);
            } else {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 1>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op, stream);
            }
            return true;
        case 3:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 3>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 4:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 4>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 5:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 5>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 6:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 6>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 7:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 7>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 8:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 8>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        default:
            return false;
    }
}

// ---- DEBUG EXPORTS (standalone kernel-diff test; remove before merge) ----
extern "C" __declspec(dllexport) void ggml_debug_pq2_0_run(const void * vx, const void * vy,
        float * dst, int ncols, int nrows, int which, uintptr_t queue_ptr) {
    dpct::queue_ptr stream = (dpct::queue_ptr) queue_ptr;
    if (which == 0) {
        mul_mat_vec_pq2_0_q8_1_v3_sycl(vx, vy, dst, ncols, nrows, stream);
    } else if (which == 2) {
        mul_mat_vec_pq2_0_q8_1_v4_sycl(vx, vy, dst, ncols, nrows, stream);
    } else if (which == 3) {
        mul_mat_vec_pq2_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream);   // v5 (base fn)
    } else if (which == 4) {
        mul_mat_vec_pq2_0_q8_1_sycl_v5n<2>(vx, vy, dst, ncols, nrows, ncols/QK8_1, ncols, stream);
    } else if (which == 5) {
        mul_mat_vec_pq2_0_q8_1_sycl_v6(vx, vy, dst, ncols, nrows, stream);
    } else if (which == 7) {
        mul_mat_vec_pq2_0_q8_1_sycl_v7(vx, vy, dst, ncols, nrows, stream);
    } else if (which == 8) {
        mul_mat_vec_pq2_0_q8_1_sycl_v8(vx, vy, dst, ncols, nrows, stream);
    } else if (which == 9) {
        mul_mat_vec_pq2_0_q8_1_sycl_v9(vx, vy, dst, ncols, nrows, stream);
    } else if (which == 10) {
        mul_mat_vec_pq2_0_q8_1_sycl_v10(vx, (const float *) vy, dst, ncols, nrows, stream);
    } else if (which == 11) {
        // 2-col test: y holds two back-to-back cols, dst two back-to-back col buffers
        mul_mat_vec_pq2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows,
                                             ncols / QK8_1, nrows, stream);
    } else {
        const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
        const sycl::range<3> block_nums(1, 1, block_num_y);
        const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_PQ2_0, QI_PQ2_0, block_pq2_0,
                                  VDR_PQ2_0_Q8_1_MMVQ, vec_dot_pq2_0_q8_1>(
                        vx, vy, dst, ncols, nrows, item_ct1);
                });
        });
    }
}
