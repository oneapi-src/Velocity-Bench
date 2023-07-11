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

// TODO: add copyright

/*
    Compute unnormalized attractive force for barnes-hut approximation of t-SNE.

    Attractive force is given by pij*qij.
*/

#include "hip/hip_runtime.h"
#include "include/kernels/attr_forces.h"

__global__ void ComputePijxQijKernelV3(
          float* __restrict__ workspace_x,
          float* __restrict__ workspace_y,
    const float* __restrict__ pij,
    const int*   __restrict__ pij_ind,
    const float* __restrict__ points,
    const int num_points,
    const int num_neighbors)
{
    int tid, i, j;
    float ix, iy, jx, jy, dx, dy, pijqij;
    tid = threadIdx.x + blockIdx.x * blockDim.x; // This is the location in the pij matrix
    if (tid >= num_points * num_neighbors)
        return;

    i = tid / num_neighbors;
    j = pij_ind[tid];

    ix = points[i];
    iy = points[num_points + i];
    jx = points[j];
    jy = points[num_points + j];
    dx = ix - jx; // X distance
    dy = iy - jy; // Y distance
    pijqij = pij[tid] / (1 + dx * dx + dy * dy);

    workspace_x[tid] = pijqij * dx;
    workspace_y[tid] = pijqij * dy;
}

__global__ void reduce_sum_kernel(
          float* __restrict__ attractive_forces,
    const float* __restrict__ workspace_x,
    const float* __restrict__ workspace_y,
    const int num_points,
    const int num_neighbors)
{
    int tid, jend, j;
    float acc_x, acc_y;
    tid = threadIdx.x + blockIdx.x * blockDim.x; // This is the location in the pij matrix
    if (tid >= num_points)
        return;

    acc_x = 0.0f;
    acc_y = 0.0f;
    jend = (tid + 1) * num_neighbors;
    for (j = tid * num_neighbors; j < jend; j++)
    {
        acc_x += workspace_x[j];
        acc_y += workspace_y[j];
    }

    attractive_forces[tid] = acc_x;
    attractive_forces[num_points + tid] = acc_y;
}

void tsnecuda::ComputeAttractiveForcesV3(
    thrust::device_vector<float>& attractive_forces,
    thrust::device_vector<float>& pij_device,
    thrust::device_vector<int>&   pij_indices_device,
    thrust::device_vector<float>& pij_workspace_device,
    thrust::device_vector<float>& points_device,
    thrust::device_vector<float>& ones_vec,
    const int num_points,
    const int num_neighbors)
{
    const int BLOCKSIZE1 = 512; // 1024;
    const int NBLOCKS1 = iDivUp(num_points * num_neighbors, BLOCKSIZE1);

    hipLaunchKernelGGL(ComputePijxQijKernelV3, NBLOCKS1, BLOCKSIZE1, 0, 0, 
        thrust::raw_pointer_cast(pij_workspace_device.data()),                              // workspace x
        thrust::raw_pointer_cast(pij_workspace_device.data()) + num_points * num_neighbors, // workspace y
        thrust::raw_pointer_cast(pij_device.data()),
        thrust::raw_pointer_cast(pij_indices_device.data()),
        thrust::raw_pointer_cast(points_device.data()),
        num_points,
        num_neighbors
    );
    HIP_CHECK_LAST_ERROR()
    GpuErrorCheck(hipDeviceSynchronize());

    const int BLOCKSIZE2 = 512;
    const int NBLOCKS2 = iDivUp(num_points, BLOCKSIZE2);

    hipLaunchKernelGGL(reduce_sum_kernel, NBLOCKS2, BLOCKSIZE2, 0, 0, 
        thrust::raw_pointer_cast(attractive_forces.data()),
        thrust::raw_pointer_cast(pij_workspace_device.data()),                              // workspace x
        thrust::raw_pointer_cast(pij_workspace_device.data()) + num_points * num_neighbors, // workspace y
        num_points,
        num_neighbors
    );
    HIP_CHECK_LAST_ERROR()
    // GpuErrorCheck(hipDeviceSynchronize());
}
