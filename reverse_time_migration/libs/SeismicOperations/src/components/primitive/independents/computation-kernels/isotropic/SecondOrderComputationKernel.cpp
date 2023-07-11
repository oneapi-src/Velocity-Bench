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

#include <operations/components/independents/concrete/computation-kernels/isotropic/SecondOrderComputationKernel.hpp>

#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>
#include <operations/exceptions/Exceptions.h>

#include <timer/Timer.h>

#include <iostream>
#include <cmath>

#define fma(a, b, c) (a) * (b) + (c)

using namespace std;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;


SecondOrderComputationKernel::SecondOrderComputationKernel(
        operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mpMemoryHandler = new WaveFieldsMemoryHandler(apConfigurationMap);
    this->mpBoundaryManager = nullptr;
    this->mCoeffXYZ = 0.0f;
}

SecondOrderComputationKernel::SecondOrderComputationKernel(
        const SecondOrderComputationKernel &aSecondOrderComputationKernel) {
    this->mpConfigurationMap = aSecondOrderComputationKernel.mpConfigurationMap;
    this->mpMemoryHandler = new WaveFieldsMemoryHandler(this->mpConfigurationMap);
    this->mpBoundaryManager = nullptr;
    this->mCoeffXYZ = 0.0f;
}

SecondOrderComputationKernel::~SecondOrderComputationKernel() = default;

void SecondOrderComputationKernel::AcquireConfiguration() {}

ComputationKernel *SecondOrderComputationKernel::Clone() {
    return new SecondOrderComputationKernel(*this);
}

void SecondOrderComputationKernel::Step() {
    /* Take a step in time. */
    if (this->mpCoeffX == nullptr) {
        this->InitializeVariables();
    }

    if ((this->mpGridBox->GetLogicalGridSize(Y_AXIS)) == 1) {
        switch (this->mpParameters->GetHalfLength()) {
            case O_2:
                this->Compute<true, O_2>();
                break;
            case O_4:
                this->Compute<true, O_4>();
                break;
            case O_8:
                this->Compute<true, O_8>();
                break;
            case O_12:
                this->Compute<true, O_12>();
                break;
            case O_16:
                this->Compute<true, O_16>();
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
    // Swap pointers : Next to current, current to prev and unwanted prev to next
    // to be overwritten.
    if (this->mpGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z) ==
        this->mpGridBox->Get(WAVE | GB_PRSS | NEXT | DIR_Z)) {
        // two pointers case : curr becomes both next and prev, while next becomes
        // current.
        this->mpGridBox->Swap(WAVE | GB_PRSS | PREV | DIR_Z, WAVE | GB_PRSS | CURR | DIR_Z);
        this->mpGridBox->Set(WAVE | GB_PRSS | NEXT | DIR_Z,
                             this->mpGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z));
    } else {
        // three pointers : normal swapping between the three pointers.
        this->mpGridBox->Swap(WAVE | GB_PRSS | PREV | DIR_Z, WAVE | GB_PRSS | CURR | DIR_Z);
        this->mpGridBox->Swap(WAVE | GB_PRSS | CURR | DIR_Z, WAVE | GB_PRSS | NEXT | DIR_Z);
    }

#ifdef ENABLE_GPU_TIMINGS
    Timer *timer = Timer::GetInstance();
    timer->StartTimer("BoundaryManager::ApplyBoundary");
#endif
    if (this->mpBoundaryManager != nullptr) {
        this->mpBoundaryManager->ApplyBoundary();
    }
#ifdef ENABLE_GPU_TIMINGS
    timer->StopTimer("BoundaryManager::ApplyBoundary");
#endif
}

void SecondOrderComputationKernel::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void SecondOrderComputationKernel::SetGridBox(GridBox *apGridBox) {
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

MemoryHandler *SecondOrderComputationKernel::GetMemoryHandler() {
    return this->mpMemoryHandler;
}

void SecondOrderComputationKernel::InitializeVariables() {
    int wnx = this->mpGridBox->GetActualWindowSize(X_AXIS);

    float dx2 = 1 / (this->mpGridBox->GetCellDimensions(X_AXIS) * this->mpGridBox->GetCellDimensions(X_AXIS));
    float dz2 = 1 / (this->mpGridBox->GetCellDimensions(Z_AXIS) * this->mpGridBox->GetCellDimensions(Z_AXIS));

    float *coeff = this->mpParameters->GetSecondDerivativeFDCoefficient();

    int hl = this->mpParameters->GetHalfLength();
    int array_length = hl;
    float coeff_x[hl];
    float coeff_z[hl];
    int vertical[hl];

    for (int i = 0; i < hl; i++) {
        coeff_x[i] = coeff[i + 1] * dx2;
        coeff_z[i] = coeff[i + 1] * dz2;

        vertical[i] = (i + 1) * (wnx);
    }

    this->mpCoeffX = new FrameBuffer<float>(array_length);
    this->mpCoeffZ = new FrameBuffer<float>(array_length);
    this->mpVerticalIdx = new FrameBuffer<int>(array_length);

    this->mCoeffXYZ = coeff[0] * (dx2 + dz2);

    Device::MemCpy(this->mpCoeffX->GetNativePointer(),      coeff_x,  array_length * sizeof(float), Device::COPY_HOST_TO_DEVICE);
    Device::MemCpy(this->mpCoeffZ->GetNativePointer(),      coeff_z,  array_length * sizeof(float), Device::COPY_HOST_TO_DEVICE);
    Device::MemCpy(this->mpVerticalIdx->GetNativePointer(), vertical, array_length * sizeof(int),   Device::COPY_HOST_TO_DEVICE);
}
