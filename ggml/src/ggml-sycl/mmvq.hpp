//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_MMVQ_HPP
#define GGML_SYCL_MMVQ_HPP

#include "common.hpp"


void ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
bool ggml_sycl_mul_mat_vec_q_id(
    enum ggml_type     src0_type,
    const void *       vx_base,             // start of stacked expert weights
    const void *       vy,                  // pre-quantized src1 (Q8_1)
    const int32_t *    ids_dev,             // device-side int32, length n_experts_used
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride, // bytes between experts in vx_base
    size_t             dst_row_stride,       // bytes between dst rows
    size_t             src1_row_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::queue_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
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
    dpct::queue_ptr    stream);

// Inverse of reorder_qw_pq2_0: restore a PQ2_0 weight tensor from SoA back to the
// standard AoS block layout, in place. Used when a multi-column (prefill) mul_mat
// runs after decode reordered the weights.
bool ggml_sycl_reorder_qw_pq2_0_restore(void * data_device, int ncols, int nrows, size_t size, size_t offset,
                                        dpct::queue_ptr stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
bool ggml_sycl_mul_mat_vec_q_glu_reorder(
    enum ggml_type     src0_type,
    enum ggml_glu_op   glu_op,
    const void *       vx,
    const void *       vgate,
    const void *       vy,
    float *            dst,
    int                ncols,                // K, shared by both weights
    int                nrows,                // output rows, i.e. weight ne[1]
    int                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    int                stride_col_y_bytes,   // bytes between activation columns in vy
    int                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);

// Batched QKV projection GEMV: three PQ2_0 weight tensors, one shared q8_1 activation,
// one kernel launch. Row blocks [0,nq)/[nq,nq+nk)/[nq+nk,nq+nk+nv) read wq/wk/wv and
// write dq/dk/dv respectively. Standard (non-reorder) block layout.
void mul_mat_vec_pq2_0_batched_sycl_v12(
    const void * vq, const void * vk, const void * vv,
    const void * vy, float * dq, float * dk, float * dv,
    int ncols, int nq, int nk, int nv,
    dpct::queue_ptr stream);

// Batched FFN gate+up GEMV: two PQ2_0 weight tensors, one shared q8_1 activation,
// one kernel launch. Row blocks [0,ng)/[ng,ng+nu) read wg/wu and write dg/du.
// Standard (non-reorder) block layout, v9 inner body.
void mul_mat_vec_pq2_0_batched_sycl_v13(
    const void * vg, const void * vu,
    const void * vy, float * dg, float * du,
    int ncols, int ng, int nu,
    dpct::queue_ptr stream);

// Batched FFN gate+up GEMV with fused swiglu epilogue: one warp computes the gate and
// up rows for the same output index and writes silu(g)*u. Gate/up ne[1] must match.
// Standard (non-reorder) block layout, v9 inner body.
void mul_mat_vec_pq2_0_batched_sycl_v14(
    const void * vg, const void * vu,
    const void * vy, float * dglu,
    int ncols, int nglu,
    dpct::queue_ptr stream);

// Diag variant: writes raw [tg, tu] pairs (2 floats/row) for epilogue bisection.
void mul_mat_vec_pq2_0_batched_sycl_v14d(
    const void * vg, const void * vu,
    const void * vy, float * dout,
    int ncols, int nglu,
    dpct::queue_ptr stream);

// v14 + also writes the raw gate/up results (v13's buffer side effects) —
// bisects whether corruption comes from leaving ngate/nup unwritten.
void mul_mat_vec_pq2_0_batched_sycl_v14w(
    const void * vg, const void * vu,
    const void * vy, float * dglu, float * dgate, float * dup,
    int ncols, int nglu,
    dpct::queue_ptr stream);

#endif // GGML_SYCL_MMVQ_HPP
