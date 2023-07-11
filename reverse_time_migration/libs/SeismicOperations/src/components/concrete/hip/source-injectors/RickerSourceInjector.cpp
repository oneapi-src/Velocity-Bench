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
// Created by amr-nasr on 13/11/2019.
//

#include <operations/components/independents/concrete/source-injectors/RickerSourceInjector.hpp>
/////#include <operations/backend/OneAPIBackend.hpp>
#include <iostream>
#include <cmath>
#include <cassert>

#include "Logging.h"

/////using namespace cl::sycl;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;
/////using namespace operations::backend;

/**
 * Implementation based on
 * https://tel.archives-ouvertes.fr/tel-00954506v2/document .
 */

__global__ void cuApplySource(float       *pressure,
                              float       *win_vel,
                              float const  ricker,
                              int   const  location)
{

    pressure[location] += (ricker * win_vel[location]);
}

void RickerSourceInjector::ApplySource(uint time_step) {
    float dt = mpGridBox->GetDT();
    float freq = mpParameters->GetSourceFrequency();

    int location = this->GetInjectionLocation();

    if (time_step < this->GetCutOffTimeStep()) {
        {
            float temp = M_PI * M_PI * freq * freq *
                         (((time_step - 1) * dt) - 1 / freq) *
                         (((time_step - 1) * dt) - 1 / freq);
            float ricker = (2 * temp - 1) * exp(-temp);

            hipLaunchKernelGGL(cuApplySource, (1,1,1), (1,1,1), 0, 0, mpGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer(), // ;grid->pressure_current, 
                                               mpGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer(), ///;grid->window_velocity, 
                                               ricker, 
                                               location);


            checkLastHIPError();
        }
    }
}
