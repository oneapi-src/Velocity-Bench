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
#include <operations/utils/interpolation/Interpolator.hpp>
#include <cassert>

using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;

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

    std::cout << "BinaryTraceManager::ApplyTraces is not implemented" << std::endl; 
    assert(0);
}
