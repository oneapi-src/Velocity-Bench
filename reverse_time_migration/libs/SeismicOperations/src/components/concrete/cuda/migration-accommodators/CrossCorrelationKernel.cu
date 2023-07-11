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


/*
 * CrossCorrelationKernel.cpp
 *
 *  Created on: Nov 29, 2020
 *      Author: aayyad
 */

#include "operations/components/independents/concrete/migration-accommodators/CrossCorrelationKernel.hpp"
/////#include <operations/backend/OneAPIBackend.hpp>

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>

#include "Logging.h"


#define EPSILON 1e-20

/////using namespace cl::sycl;
using namespace std;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;
/////using namespace operations::backend;

template void CrossCorrelationKernel::Correlation<true, NO_COMPENSATION>(GridBox *apGridBox);

template void CrossCorrelationKernel::Correlation<false, NO_COMPENSATION>(GridBox *apGridBox);

template void CrossCorrelationKernel::Correlation<true, COMBINED_COMPENSATION>(GridBox *apGridBox);

template void CrossCorrelationKernel::Correlation<false, COMBINED_COMPENSATION>(GridBox *apGridBox);

__global__ void cuCorrelationKernel(float                   *output_buffer,
                                    float                   *src_buffer,
                                    float                   *dest_buffer,
                                    float                   *source,
                                    float                   *receiver,
                                    COMPENSATION_TYPE const _COMPENSATION_TYPE,
                                    int               const half_length_offset,
                                    int               const y_offset,
                                    int               const wnx,
                                    int               const wnz
        )
{
    int const iX = blockIdx.x * blockDim.x + threadIdx.x + half_length_offset;
    int const iY = blockIdx.y * blockDim.y + threadIdx.y + half_length_offset;
    int const iZ = blockIdx.z * blockDim.z + threadIdx.z + y_offset;

    int const idx = iZ * wnz * wnx + (iY) * wnx + (iX);
    output_buffer[idx] += source[idx] * receiver[idx];

    if (_COMPENSATION_TYPE == COMPENSATION_TYPE::COMBINED_COMPENSATION) {
        src_buffer[idx] += source[idx] * source[idx];
        dest_buffer[idx] += receiver[idx] * receiver[idx];
    }
}


template<bool _IS_2D, COMPENSATION_TYPE _COMPENSATION_TYPE>
void CrossCorrelationKernel::Correlation(GridBox *apGridBox) {

    int wnx = apGridBox->GetActualWindowSize(X_AXIS);
    int wny = apGridBox->GetActualWindowSize(Y_AXIS);
    int wnz = apGridBox->GetActualWindowSize(Z_AXIS);

    int compute_nx = apGridBox->GetComputationGridSize(X_AXIS);
    int compute_ny = apGridBox->GetComputationGridSize(Y_AXIS);
    int compute_nz = apGridBox->GetComputationGridSize(Z_AXIS);

    int block_x = mpParameters->GetBlockX();
    int block_y = mpParameters->GetBlockY();
    int block_z = mpParameters->GetBlockZ();
    int half_length = mpParameters->GetHalfLength();

    int y_offset = half_length;
    if (_IS_2D) {
        y_offset = 0;
    }

    float *source = apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    float *receiver = mpGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();

    float *output_buffer = mpShotCorrelation->GetNativePointer();
    float *src_buffer = mpSourceIllumination->GetNativePointer();
    float *dest_buffer = mpReceiverIllumination->GetNativePointer();

    dim3 const cuCorrelationBlockSize(block_x, block_z, block_y), cuCorrelationGridSize(compute_nx / block_x, compute_nz / block_z, compute_ny);

    cuCorrelationKernel<<<cuCorrelationGridSize, cuCorrelationBlockSize>>>
                       (output_buffer,
                        src_buffer,
                        dest_buffer,
                        source,
                        receiver,
                        _COMPENSATION_TYPE,
                        half_length,
                        y_offset,
                        wnx,
                        wnz
        );

    checkLastCUDAError();
}

__global__ void cuStack(float *pTotalCorrelation, 
                        float *pShotCorrelation,
                        float *pTotalSourceIllumination,
                        float *pSourceIllumination,
                        float *pTotalReceiverIllumination,
                        float *pReceiverIllumination,
                        int const nx,
                        int const nz,
                        int const wsx,
                        int const wsy,
                        int const wsz,
                        int const wnx,
                        int const wnz,
                        COMPENSATION_TYPE const CompensationType 
        )
{
    unsigned int const idx0 = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int const idx1 = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int const idx2 = blockDim.z * blockIdx.z + threadIdx.z;

    float *stack_buf = pTotalCorrelation + wsx + wsz * nx + wsy * nx * nz;
    float *cor_buf   = pShotCorrelation;
    float *stack_src = pTotalSourceIllumination + wsx + wsz * nx + wsy * nx * nz;
    float *cor_src   = pSourceIllumination;
    float *stack_rcv = pTotalReceiverIllumination + wsx + wsz * nx + wsy * nx * nz;
    float *cor_rcv   = pReceiverIllumination;
    if (CompensationType == NO_COMPENSATION) {
        uint offset_window = idx0 + idx1 * wnx + idx2 * wnx * wnz;
        uint offset = idx0 + idx1 * nx + idx2 * nx * nz;
        stack_buf[offset] += cor_buf[offset_window];
    } else {
        uint offset_window = idx0 + idx1 * wnx + idx2 * wnx * wnz;
        uint offset = idx0 + idx1 * nx + idx2 * nx * nz;
        stack_buf[offset] += cor_buf[offset_window];
        stack_src[offset] += cor_src[offset_window];
        stack_rcv[offset] += cor_rcv[offset_window];
    }
}

void CrossCorrelationKernel::Stack() 
{
    int nx = mpGridBox->GetActualGridSize(X_AXIS);
    int ny = mpGridBox->GetActualGridSize(Y_AXIS);
    int nz = mpGridBox->GetActualGridSize(Z_AXIS);
    int wnx = mpGridBox->GetActualWindowSize(X_AXIS);
    int wny = mpGridBox->GetActualWindowSize(Y_AXIS);
    int wnz = mpGridBox->GetActualWindowSize(Z_AXIS);
    int orig_x = mpGridBox->GetLogicalWindowSize(X_AXIS);
    int orig_y = mpGridBox->GetLogicalWindowSize(Y_AXIS);
    int orig_z = mpGridBox->GetLogicalWindowSize(Z_AXIS);
    int wsx = mpGridBox->GetWindowStart(X_AXIS);
    int wsz = mpGridBox->GetWindowStart(Z_AXIS);
    int wsy = mpGridBox->GetWindowStart(Y_AXIS);

    size_t sizeTotal = nx * nz * ny;

    dim3 const cuBlockSizeStack(1, 1, 1), cuGridSizeStack(orig_x, orig_z, orig_y);
    cuStack<<<cuGridSizeStack, cuBlockSizeStack>>>
                       (mpTotalCorrelation->GetNativePointer(), 
                        mpShotCorrelation->GetNativePointer(),
                        mpTotalSourceIllumination->GetNativePointer(),
                        mpSourceIllumination->GetNativePointer(),
                        mpTotalReceiverIllumination->GetNativePointer(),
                        mpReceiverIllumination->GetNativePointer(),
                        nx,
                        nz,
                        wsx,
                        wsy,
                        wsz,
                        wnx,
                        wnz,
                        mCompensationType); 


    cudaDeviceSynchronize();
    checkLastCUDAError();
}

