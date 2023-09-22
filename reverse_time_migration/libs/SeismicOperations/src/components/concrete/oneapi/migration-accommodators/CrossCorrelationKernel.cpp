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
#include <operations/backend/OneAPIBackend.hpp>

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>

#define EPSILON 1e-20

using namespace sycl;
using namespace std;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;
using namespace operations::backend;

template void CrossCorrelationKernel::Correlation<true, NO_COMPENSATION>(GridBox *apGridBox);

template void CrossCorrelationKernel::Correlation<false, NO_COMPENSATION>(GridBox *apGridBox);

template void CrossCorrelationKernel::Correlation<true, COMBINED_COMPENSATION>(GridBox *apGridBox);

template void CrossCorrelationKernel::Correlation<false, COMBINED_COMPENSATION>(GridBox *apGridBox);

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

    OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {

        auto global_range = range<2>(compute_nz, compute_nx);
        auto local_range = range<2>(block_z, block_x);
        auto starting_offset = id<2>(half_length, half_length);
        auto global_nd_range = nd_range<2>(global_range,
                                           local_range);
                                           ///starting_offset);

        float *output_buffer = mpShotCorrelation->GetNativePointer();
        float *src_buffer = mpSourceIllumination->GetNativePointer();
        float *dest_buffer = mpReceiverIllumination->GetNativePointer();
        cgh.parallel_for(global_nd_range, [=](nd_item<2> it)  {

            int idx = (it.get_global_id(0) * wnx      ) + (starting_offset[0] * wnx) +
                      it.get_global_id(1)               +  starting_offset[1];

            output_buffer[idx] += source[idx] * receiver[idx];

            if (_COMPENSATION_TYPE == COMPENSATION_TYPE::COMBINED_COMPENSATION) {
                src_buffer[idx] += source[idx] * source[idx];
                dest_buffer[idx] += receiver[idx] * receiver[idx];
            }

        });
    });

    ////OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
}

void CrossCorrelationKernel::Stack() {
    int nx = mpGridBox->GetActualGridSize(X_AXIS);
    int ny = mpGridBox->GetActualGridSize(Y_AXIS);
    int nz = mpGridBox->GetActualGridSize(Z_AXIS);
    int wnx = mpGridBox->GetActualWindowSize(X_AXIS);
    int wny = mpGridBox->GetActualWindowSize(Y_AXIS);
    int wnz = mpGridBox->GetActualWindowSize(Z_AXIS);
    int orig_x = mpGridBox->GetLogicalWindowSize(X_AXIS);
    int orig_y = mpGridBox->GetLogicalWindowSize(Y_AXIS);
    int orig_z = mpGridBox->GetLogicalWindowSize(Z_AXIS);

    size_t sizeTotal = nx * nz * ny;
    OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
        auto global_range = range<2>(orig_z, orig_x); 
        auto local_range  = sycl::range<2>(1, 1);
        auto global_nd_range = sycl::nd_range<2>(global_range, local_range);
        int wsx = mpGridBox->GetWindowStart(X_AXIS);
        int wsz = mpGridBox->GetWindowStart(Z_AXIS);
        int wsy = mpGridBox->GetWindowStart(Y_AXIS);
        float *stack_buf = mpTotalCorrelation->GetNativePointer() + wsx + wsz * nx + wsy * nx * nz;
        float *cor_buf = mpShotCorrelation->GetNativePointer();
        float *stack_src = mpTotalSourceIllumination->GetNativePointer() + wsx + wsz * nx + wsy * nx * nz;
        float *cor_src = mpSourceIllumination->GetNativePointer();
        float *stack_rcv = mpTotalReceiverIllumination->GetNativePointer() + wsx + wsz * nx + wsy * nx * nz;
        float *cor_rcv = mpReceiverIllumination->GetNativePointer();
        if (mCompensationType == NO_COMPENSATION) {
            cgh.parallel_for(
                    global_nd_range, [=](sycl::nd_item<2> it) {
                        uint offset_window = it.get_global_id(0) * wnx + it.get_global_id(1); 
                        uint offset = it.get_global_id(0) * nx + it.get_global_id(1); 
                        stack_buf[offset] += cor_buf[offset_window];
                    });
        } else {
            cgh.parallel_for(
                    global_nd_range, [=](sycl::id<2> idx) {
                        uint offset_window = idx[0] + idx[1] * wnx;
                        uint offset = idx[0] + idx[1] * nx;
                        stack_buf[offset] += cor_buf[offset_window];
                        stack_src[offset] += cor_src[offset_window];
                        stack_rcv[offset] += cor_rcv[offset_window];
                    });
        }
    });
    OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
}

