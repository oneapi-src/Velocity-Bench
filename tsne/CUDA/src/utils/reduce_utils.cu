/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @brief Utilities for different reductions
 * 
 * @file reduce_utils.cu
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#include "include/utils/reduce_utils.h"

// expects matrix of size N x M
thrust::device_vector<float> tsnecuda::utils::ReduceAlpha(
    cublasHandle_t& handle,
    const thrust::device_vector<float>& d_matrix,
    const int N,
    const int M,
    float alpha,
    const int axis)
{
    if (axis == 0) {
        thrust::device_vector<float> ones(N, 1.f);
        thrust::device_vector<float> means(M);

        float kBeta = 0.f;
        CublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha,
                thrust::raw_pointer_cast(d_matrix.data()), N,
                thrust::raw_pointer_cast(ones.data()), 1, &kBeta,
                thrust::raw_pointer_cast(means.data()), 1));
        GpuErrorCheck(cudaDeviceSynchronize());
        return means;
    } else if (axis == 1) {
        thrust::device_vector<float> ones(M, 1.f);
        thrust::device_vector<float> means(N);

        float kBeta = 0.f;
        CublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha,
                thrust::raw_pointer_cast(d_matrix.data()), N,
                thrust::raw_pointer_cast(ones.data()), 1, &kBeta,
                thrust::raw_pointer_cast(means.data()), 1));
        GpuErrorCheck(cudaDeviceSynchronize());
        return means;
    } else {
        throw std::runtime_error("Axis must be 0 or 1.");
    }
}

thrust::device_vector<float> tsnecuda::utils::ReduceSum(
    cublasHandle_t &handle,
    const thrust::device_vector<float>& d_matrix,
    const int N,
    const int M,
    const int axis)
{
    float alpha = 1.f;
    return tsnecuda::utils::ReduceAlpha(handle, d_matrix, N, M, alpha, axis);
}
