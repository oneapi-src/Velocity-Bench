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
 * @brief Implementation of the math_utils.h file
 *
 * @file math_utils.cu
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#include "include/utils/math_utils.h"

float tsnecuda::utils::L2NormDeviceVector(
    const thrust::device_vector<float>& d_vector)
{
    return std::sqrt(thrust::transform_reduce(
        d_vector.begin(),
        d_vector.end(),
        tsnecuda::utils::FunctionalSquare(),
        0.0f,
        thrust::plus<float>()));
}

void tsnecuda::utils::MaxNormalizeDeviceVector(
    thrust::device_vector<float>& d_vector)
{
    float max_val = thrust::transform_reduce(d_vector.begin(), d_vector.end(), tsnecuda::utils::FunctionalAbs(), 0.0f, thrust::maximum<float>());

    thrust::constant_iterator<float> division_iterator(max_val);
    thrust::transform(d_vector.begin(), d_vector.end(), division_iterator, d_vector.begin(), thrust::divides<float>());
}

__global__
void syv2k(
          float* __restrict__ pij_sym,
    const float* __restrict__ pij_non_sym,
    const int*   __restrict__ pij_indices,
    const int num_points,
    const int num_neighbors)
{
    int tid, i, j, jend;
    float pij_acc;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points * num_neighbors) {
        return;
    }

    i = tid / num_neighbors;
    j = pij_indices[tid];

    pij_acc = pij_non_sym[tid];
    jend = (j + 1) * num_neighbors;
    for (int jidx = j * num_neighbors; jidx < jend; jidx++) {
        pij_acc += pij_indices[jidx] == i ? pij_non_sym[jidx] : 0.0f;
    }
    pij_sym[tid] = pij_acc / (2.0f * num_points);
}

void tsnecuda::utils::SymmetrizeMatrixV2(
    thrust::device_vector<float>&   pij_symmetric,     // output
    thrust::device_vector<float>&   pij_nonsymmetric,  // input
    thrust::device_vector<int32_t>& pij_indices,       // input
    const int num_points,
    const int num_neighbors)
{
    const int num_threads = 32; //1024;
    const int num_blocks = iDivUp(num_points * num_neighbors, num_threads);

    syv2k<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(pij_symmetric.data()),
        thrust::raw_pointer_cast(pij_nonsymmetric.data()),
        thrust::raw_pointer_cast(pij_indices.data()),
        num_points,
        num_neighbors
    );
    CUDA_CHECK_LAST_ERROR()
    GpuErrorCheck(cudaDeviceSynchronize());
}
