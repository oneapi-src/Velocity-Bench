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


//
// Created by ingy-mounir on 1/28/20.
//

#include "operations/components/independents/concrete/trace-managers/SeismicTraceManager.hpp"
//////#include <operations/backend/OneAPIBackend.hpp>
#include <cassert>
#include <fstream>


#include "Logging.h"

//////using namespace cl::sycl;
using namespace std;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;
///////using namespace operations::backend;

__global__ void cuApplyTraces(float *current,
                              float *trace_values,
                              float *w_vel,
                              uint *x_pos,
                              uint *y_pos,
                              int const wnz_wnx,
                              int const std_offset,
                              uint const trace_step,
                              int const  trace_size
        )
{
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y * blockDim.y + threadIdx.y;
    int const z = blockIdx.z * blockDim.z + threadIdx.z;

    int const blockId  = (gridDim.x * gridDim.y * blockIdx.z) + (gridDim.x * blockIdx.y) + blockIdx.x;
    int const i = (blockId * blockDim.x) + threadIdx.x;

    int offset = y_pos[i] * wnz_wnx + std_offset + x_pos[i];
    current[offset] += trace_values[(trace_step)*trace_size + i] * w_vel[offset];
}


void SeismicTraceManager::ApplyTraces(uint time_step) {
    int trace_size = mpTracesHolder->TraceSizePerTimeStep;
    int wnx = mpGridBox->GetActualWindowSize(X_AXIS);
    int wnz_wnx = mpGridBox->GetActualWindowSize(Z_AXIS) * wnx;
    int std_offset = (mpParameters->GetBoundaryLength() + mpParameters->GetHalfLength()) * wnx;
    float current_time = (time_step - 1) * mpGridBox->GetDT();
    float dt = mpGridBox->GetDT();

    uint trace_step = uint(current_time / mpTracesHolder->SampleDT);
    trace_step = std::min(trace_step, mpTracesHolder->SampleNT - 1);


    dim3 const cuBlockSize(1, 1, 1), cuGridSize(trace_size, 1, 1);
    cuApplyTraces<<<cuGridSize, cuBlockSize>>>(mpGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer(),   /// *current,
                  mpDTraces.GetNativePointer(),   ///*trace_values,
                  mpGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer(),   // *w_vel,
                  mpDPositionsX.GetNativePointer(),  /// *x_pos,
                  mpDPositionsY.GetNativePointer(),  // *y_pos,
                  wnz_wnx,
                  std_offset,
                  trace_step,
                  trace_size
        );


    checkLastCUDAError();
}
