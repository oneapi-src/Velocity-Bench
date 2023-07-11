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

#include "hip/hip_runtime.h"
#include "include/kernels/rep_forces.h"

__global__ void compute_repulsive_forces_kernel(
    volatile float* __restrict__ repulsive_forces,
    volatile float* __restrict__ normalization_vec,
    const float* const xs,
    const float* const ys,
    const float* const potentialsQij,
    const int num_points,
    const int n_terms)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points)
        return;

    float phi1, phi2, phi3, phi4, x_pt, y_pt;

    phi1 = potentialsQij[tid * n_terms + 0];
    phi2 = potentialsQij[tid * n_terms + 1];
    phi3 = potentialsQij[tid * n_terms + 2];
    phi4 = potentialsQij[tid * n_terms + 3];

    x_pt = xs[tid];
    y_pt = ys[tid];

    normalization_vec[tid] = (1 + x_pt * x_pt + y_pt * y_pt) * phi1 - 2 * (x_pt * phi2 + y_pt * phi3) + phi4;

    repulsive_forces[tid] = x_pt * phi1 - phi2;
    repulsive_forces[tid + num_points] = y_pt * phi1 - phi3;
}

float tsnecuda::ComputeRepulsiveForces(
    thrust::device_vector<float>& repulsive_forces,
    thrust::device_vector<float>& normalization_vec,
    thrust::device_vector<float>& points_device,
    thrust::device_vector<float>& potentialsQij,
    const int num_points,
    const int n_terms)
{
    const int BLOCKSIZE = 512; //1024;
    const int NBLOCKS = (num_points + BLOCKSIZE - 1) / BLOCKSIZE;

    hipLaunchKernelGGL(compute_repulsive_forces_kernel, NBLOCKS, BLOCKSIZE, 0, 0, 
        thrust::raw_pointer_cast(repulsive_forces.data()),
        thrust::raw_pointer_cast(normalization_vec.data()),
        thrust::raw_pointer_cast(points_device.data()),
        thrust::raw_pointer_cast(points_device.data() + num_points),
        thrust::raw_pointer_cast(potentialsQij.data()),
        num_points,
        n_terms
    );
    HIP_CHECK_LAST_ERROR()
    GpuErrorCheck(hipDeviceSynchronize());

    float sumQ = thrust::reduce(
        normalization_vec.begin(), normalization_vec.end(), 0.0f,
        thrust::plus<float>());
    return sumQ - num_points;
}

__global__ void compute_chargesQij_kernel(
    volatile float* __restrict__ chargesQij,
    const float* const xs,
    const float* const ys,
    const int num_points,
    const int n_terms)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points)
        return;

    float x_pt, y_pt;
    x_pt = xs[tid];
    y_pt = ys[tid];

    chargesQij[tid * n_terms + 0] = 1;
    chargesQij[tid * n_terms + 1] = x_pt;
    chargesQij[tid * n_terms + 2] = y_pt;
    chargesQij[tid * n_terms + 3] = x_pt * x_pt + y_pt * y_pt;
}

void tsnecuda::ComputeChargesQij(
    thrust::device_vector<float>& chargesQij,
    thrust::device_vector<float>& points_device,
    const int num_points,
    const int n_terms)
{
    const int BLOCKSIZE = 512; //1024;
    const int NBLOCKS = (num_points + BLOCKSIZE - 1) / BLOCKSIZE;

    hipLaunchKernelGGL(compute_chargesQij_kernel, NBLOCKS, BLOCKSIZE, 0, 0, 
        thrust::raw_pointer_cast(chargesQij.data()),
        thrust::raw_pointer_cast(points_device.data()),
        thrust::raw_pointer_cast(points_device.data() + num_points),
        num_points,
        n_terms
    );
    HIP_CHECK_LAST_ERROR()
    GpuErrorCheck(hipDeviceSynchronize());
}
