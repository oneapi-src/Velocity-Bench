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
// Created by amr-nasr on 13/11/2019.
//

#include <operations/components/independents/concrete/trace-managers/BinaryTraceManager.hpp>
#include <operations/backend/OneAPIBackend.hpp>
#include <operations/utils/interpolation/Interpolator.hpp>

using namespace sycl;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;
using namespace operations::backend;

void BinaryTraceManager::ApplyTraces(uint time_step) {
    int x_inc = mReceiverIncrement.x == 0 ? 1 : mReceiverIncrement.x;
    int y_inc = mReceiverIncrement.y == 0 ? 1 : mReceiverIncrement.y;
    int z_inc = mReceiverIncrement.z == 0 ? 1 : mReceiverIncrement.z;
    int trace_size = mpTracesHolder->TraceSizePerTimeStep;
    int wnx = mpGridBox->GetActualWindowSize(X_AXIS);
    int wnz_wnx = mpGridBox->GetActualWindowSize(Z_AXIS) * wnx;
    float current_time = (time_step - 1) * mpGridBox->GetDT();
    float dt = mpGridBox->GetDT();
    uint trace_step = uint(current_time / mpTracesHolder->SampleDT);
    if (trace_step > mpTracesHolder->SampleNT - 1) {
        trace_step = mpTracesHolder->SampleNT - 1;
    }
    int r_start_y = mReceiverStart.y, r_start_x = mReceiverStart.x, r_start_z = mReceiverStart.z;
    int trace_nx = mpTracesHolder->ReceiversCountX;

    OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
        auto global_range =
                range<3>(1, mpTracesHolder->ReceiversCountY, mpTracesHolder->ReceiversCountX);
        auto local_range = range<3>(1, 1, 1);
        auto global_nd_range = nd_range<3>(global_range, local_range);

        float *current = mpGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
        float *ptr_traces = mpDTraces.GetNativePointer();
        float *w_vel = mpGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer();
        cgh.parallel_for<class trace_manager>(global_nd_range, [=](nd_item<3> it) {
            int iz = it.get_global_id(0);
            int iy = it.get_global_id(1);
            int ix = it.get_global_id(2);
            int offset = ((iy * y_inc) + r_start_y) * wnz_wnx +
                         ((iz * z_inc) + r_start_z) * wnx +
                         ((ix * x_inc) + r_start_x);
            current[offset] += ptr_traces[(trace_step) * trace_size + iy * trace_nx + ix] * w_vel[offset];
        });
    });

    OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
}
