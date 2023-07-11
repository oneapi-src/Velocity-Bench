/*
 * Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU Lesser General Public License v3.0 or later
 * 
 * If a copy of the license was not distributed with this file, you can obtain one at 
 * https://www.gnu.org/licenses/lgpl-3.0-standalone.html
 * 
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


#include "hip/hip_runtime.h"
//
// Created by amr-nasr on 11/21/19.
//

#include <operations/components/independents/concrete/computation-kernels/isotropic/SecondOrderComputationKernel.hpp>

#include <timer/Timer.h>
#include <memory-manager/MemoryManager.h>

#include <cstring>
#include <cassert>

#include "Logging.h"

////#define fma(a, b, c) (a) * (b) + (c)

#define HALF_LENGTH_CUDA 8
//////using namespace cl::sycl;
using namespace std;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;

template void SecondOrderComputationKernel::Compute<true, O_2>();

template void SecondOrderComputationKernel::Compute<true, O_4>();

template void SecondOrderComputationKernel::Compute<true, O_8>();

template void SecondOrderComputationKernel::Compute<true, O_12>();

template void SecondOrderComputationKernel::Compute<true, O_16>();

template void SecondOrderComputationKernel::Compute<false, O_2>();

template void SecondOrderComputationKernel::Compute<false, O_4>();

template void SecondOrderComputationKernel::Compute<false, O_8>();

template void SecondOrderComputationKernel::Compute<false, O_12>();

template void SecondOrderComputationKernel::Compute<false, O_16>();

__global__ void ComputeKernel(float *curr_base,
                              float *prev_base,
                              float *next_base,
                              float *vel_base,
                              int const block_z,
                              float *mpCoeffX,
                              float *mpCoeffZ,
                              float const mCoeffXYZ,
                              int   *mpVerticalIdx,
                              size_t const nx

        )
{
    int const hl = HALF_LENGTH_CUDA; 
    const float *current = curr_base;
    const float *prev = prev_base;
    float *next = next_base;
    const float *vel = vel_base;
    const float *c_x = mpCoeffX; 
    const float *c_z = mpCoeffZ; 
    const float c_xyz = mCoeffXYZ;
    const int *v = mpVerticalIdx; 
    const int idx_range = block_z; 
    const int pad = 0;

    __shared__ float local[136]; // shared memory 
    int iIndexX = blockDim.x * blockIdx.x + threadIdx.x;
    int iIndexY = blockDim.y * blockIdx.y + threadIdx.y;


    int idx = iIndexX + hl + (iIndexY * idx_range + hl) * nx;
    size_t id0 = threadIdx.x;

    size_t identifiant = (id0 + hl);
    float c_x_loc[HALF_LENGTH_CUDA];
    float c_z_loc[HALF_LENGTH_CUDA];
    int v_end = v[HALF_LENGTH_CUDA - 1];
    float front[HALF_LENGTH_CUDA + 1];
    float back[HALF_LENGTH_CUDA];
    for (unsigned int iter = 0; iter <= HALF_LENGTH_CUDA; iter++) {
        front[iter] = current[idx + nx * iter];
    }
    for (unsigned int iter = 1; iter <= HALF_LENGTH_CUDA; iter++) {
        back[iter - 1] = current[idx - nx * iter];
        c_x_loc[iter - 1] = c_x[iter - 1];
        c_z_loc[iter - 1] = c_z[iter - 1];
    }
    bool copyHaloX = false;
    if (id0 < HALF_LENGTH_CUDA)
        copyHaloX = true;
    const unsigned int items_X = blockDim.x;
    for (int i = 0; i < idx_range; i++) {
        local[identifiant] = front[0];
        if (copyHaloX) {
            local[identifiant - HALF_LENGTH_CUDA] = current[idx - HALF_LENGTH_CUDA];
            local[identifiant + items_X] = current[idx + items_X];
        }
        __syncthreads();
        ////it.barrier(access::fence_space::local_space);
        float value = 0;
        value = fmaf(local[identifiant], c_xyz, value);
        for (int iter = 1; iter <= HALF_LENGTH_CUDA; iter++) {
            value = fmaf(local[identifiant - iter], c_x_loc[iter - 1], value);
            value = fmaf(local[identifiant + iter], c_x_loc[iter - 1], value);
        }
        for (int iter = 1; iter <= HALF_LENGTH_CUDA; iter++) {
            value = fmaf(front[iter], c_z_loc[iter - 1], value);
            value = fmaf(back[iter - 1], c_z_loc[iter - 1], value);
        }
        value = fmaf(vel[idx], value, -prev[idx]);
        value = fmaf(2.0f, local[identifiant], value);
        next[idx] = value;
        idx += nx;
        for (unsigned int iter = HALF_LENGTH_CUDA - 1; iter > 0; iter--) {
            back[iter] = back[iter - 1];
        }
        back[0] = front[0];
        for (unsigned int iter = 0; iter < HALF_LENGTH_CUDA; iter++) {
            front[iter] = front[iter + 1];
        }
        // Only one new data-point read from global memory
        // in z-dimension (depth)
        front[HALF_LENGTH_CUDA] = current[idx + v_end];
    }
}


template<bool IS_2D_, HALF_LENGTH HALF_LENGTH_>
void SecondOrderComputationKernel::Compute() {
    // Read parameters into local variables to be shared.

    size_t nx = mpGridBox->GetActualWindowSize(X_AXIS);
    size_t nz = mpGridBox->GetActualWindowSize(Z_AXIS);

    float *prev_base = mpGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();
    float *curr_base = mpGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    float *next_base = mpGridBox->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();

    float *vel_base = mpGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer();

    // Pre-compute the coefficients for each direction.
    int hl = HALF_LENGTH_;

    int compute_nz = mpGridBox->GetComputationGridSize(Z_AXIS) / mpParameters->GetBlockZ();
    assert(mpGridBox->GetComputationGridSize(X_AXIS) % mpParameters->GetBlockX() == 0);
    dim3 const cuBlockSize(mpParameters->GetBlockX(), 1), cuGridSize(mpGridBox->GetComputationGridSize(X_AXIS) / mpParameters->GetBlockX(), compute_nz);
    /////std::cout << "Grid Size : " << cuGridSize.x  << ", " << cuGridSize.y << std::endl;
    /////std::cout << "Block Size: " << cuBlockSize.x << ", " << cuBlockSize.y << std::endl;
 
    hipLaunchKernelGGL(ComputeKernel, cuGridSize, cuBlockSize, 0, 0,
                  curr_base,
                  prev_base,
                  next_base,
                  vel_base,
                  mpParameters->GetBlockZ(),
                  mpCoeffX->GetNativePointer(),
                  mpCoeffZ->GetNativePointer(), 
                  mCoeffXYZ,
                  mpVerticalIdx->GetNativePointer(),
                  nx);
    checkLastHIPError();
}
