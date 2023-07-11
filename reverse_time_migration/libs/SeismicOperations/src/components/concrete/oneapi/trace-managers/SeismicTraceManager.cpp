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
#include <operations/backend/OneAPIBackend.hpp>

using namespace sycl;
using namespace std;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;
using namespace operations::backend;

void SeismicTraceManager::ApplyTraces(uint time_step) {
    int trace_size = mpTracesHolder->TraceSizePerTimeStep;
    int wnx = mpGridBox->GetActualWindowSize(X_AXIS);
    int wnz_wnx = mpGridBox->GetActualWindowSize(Z_AXIS) * wnx;
    int std_offset = (mpParameters->GetBoundaryLength() + mpParameters->GetHalfLength()) * wnx;
    float current_time = (time_step - 1) * mpGridBox->GetDT();
    float dt = mpGridBox->GetDT();

    uint trace_step = uint(current_time / mpTracesHolder->SampleDT);
    trace_step = std::min(trace_step, mpTracesHolder->SampleNT - 1);

    OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
        auto global_range = range<1>(trace_size);
        auto local_range = range<1>(1);
        auto global_nd_range = nd_range<1>(global_range, local_range);

        float *current = mpGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
        float *trace_values = mpDTraces.GetNativePointer();
        float *w_vel = mpGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer();
        uint *x_pos = mpDPositionsX.GetNativePointer();
        uint *y_pos = mpDPositionsY.GetNativePointer();
        cgh.parallel_for<class trace_manager>(global_nd_range, [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            int offset = y_pos[i] * wnz_wnx + std_offset + x_pos[i];
            current[offset] += trace_values[(trace_step) * trace_size + i] * w_vel[offset];
        });
    });
    //////OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
}
