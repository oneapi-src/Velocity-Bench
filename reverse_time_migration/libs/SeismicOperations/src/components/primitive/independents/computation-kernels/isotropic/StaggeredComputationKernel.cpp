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
// Created by amr-nasr on 11/21/19.
//

#include <operations/components/independents/concrete/computation-kernels/isotropic/StaggeredComputationKernel.hpp>

#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>

#include <timer/Timer.h>

#include <iostream>
#include <cmath>

using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;


StaggeredComputationKernel::StaggeredComputationKernel(
        operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mpMemoryHandler = new WaveFieldsMemoryHandler(this->mpConfigurationMap);
    this->mpBoundaryManager = nullptr;
}

StaggeredComputationKernel::StaggeredComputationKernel(const StaggeredComputationKernel &aStaggeredComputationKernel) {
    this->mpConfigurationMap = aStaggeredComputationKernel.mpConfigurationMap;
    this->mpMemoryHandler = new WaveFieldsMemoryHandler(this->mpConfigurationMap);
    this->mpBoundaryManager = nullptr;
}

StaggeredComputationKernel::~StaggeredComputationKernel() = default;

void StaggeredComputationKernel::AcquireConfiguration() {}

ComputationKernel *StaggeredComputationKernel::Clone() {
    return new StaggeredComputationKernel(*this);
}

void StaggeredComputationKernel::Step() {
    /* Take a step in time. */
    if (this->mAdjoint) {
        if (this->mpGridBox->GetLogicalGridSize(Y_AXIS) == 1) {
            switch (this->mpParameters->GetHalfLength()) {
                case O_2:
                    Compute<true, O_2>();
                    break;
                case O_4:
                    Compute<true, O_4>();
                    break;
                case O_8:
                    Compute<true, O_8>();
                    break;
                case O_12:
                    Compute<true, O_12>();
                    break;
                case O_16:
                    Compute<true, O_16>();
                    break;
            }
        } else {
            switch (this->mpParameters->GetHalfLength()) {
                case O_2:
                    Compute<true, O_2>();
                    break;
                case O_4:
                    Compute<true, O_4>();
                    break;
                case O_8:
                    Compute<true, O_8>();
                    break;
                case O_12:
                    Compute<true, O_12>();
                    break;
                case O_16:
                    Compute<true, O_16>();
                    break;
            }
        }
    } else {
        if ((this->mpGridBox->GetLogicalGridSize(Y_AXIS)) == 1) {
            switch (this->mpParameters->GetHalfLength()) {
                case O_2:
                    this->Compute<false, O_2>();
                    break;
                case O_4:
                    this->Compute<false, O_4>();
                    break;
                case O_8:
                    this->Compute<false, O_8>();
                    break;
                case O_12:
                    this->Compute<false, O_12>();
                    break;
                case O_16:
                    this->Compute<false, O_16>();
                    break;
            }
        } else {
            switch (this->mpParameters->GetHalfLength()) {

                case O_2:
                    this->Compute<false, O_2>();
                    break;
                case O_4:
                    this->Compute<false, O_4>();
                    break;
                case O_8:
                    this->Compute<false, O_8>();
                    break;
                case O_12:
                    this->Compute<false, O_12>();
                    break;
                case O_16:
                    this->Compute<false, O_16>();
                    break;
            }
        }
    }
    this->mpGridBox->Swap(WAVE | GB_PRSS | NEXT | DIR_Z, WAVE | GB_PRSS | CURR | DIR_Z);

    Timer *timer = Timer::GetInstance();
    timer->StartTimer("BoundaryManager::ApplyBoundary(Pressure)");
    if (this->mpBoundaryManager != nullptr) {
        this->mpBoundaryManager->ApplyBoundary(0);
    }
    timer->StopTimer("BoundaryManager::ApplyBoundary(Pressure)");
}

void StaggeredComputationKernel::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void StaggeredComputationKernel::SetGridBox(GridBox *apGridBox) {
    this->mpGridBox = apGridBox;
    if (this->mpGridBox == nullptr) {
        std::cerr << "No GridBox provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    /* Does not support 3D. */
    if (this->mpGridBox->GetActualWindowSize(Y_AXIS) > 1) {
        throw exceptions::NotImplementedException();
    }
}

MemoryHandler *StaggeredComputationKernel::GetMemoryHandler() {
    return this->mpMemoryHandler;
}
