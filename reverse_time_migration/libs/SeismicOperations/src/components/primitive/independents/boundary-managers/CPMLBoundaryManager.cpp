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
// Created by amr-nasr on 18/11/2019.
//

#include <operations/components/independents/concrete/boundary-managers/CPMLBoundaryManager.hpp>

#include <operations/components/independents/concrete/boundary-managers/extensions/HomogenousExtension.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

#ifndef PWR2
#define PWR2(EXP) ((EXP) * (EXP))
#endif

#define fma(a, b, c) (((a) * (b)) + (c))

using namespace operations::components;
using namespace operations::components::addons;
using namespace operations::common;
using namespace operations::dataunits;

CPMLBoundaryManager::CPMLBoundaryManager(operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mReflectCoefficient = 0.1;
    this->mShiftRatio = 0.1;
    this->mRelaxCoefficient = 0.1;
    this->mUseTopLayer = true;
    this->mpExtension = nullptr;
    this->coeff_a_x = nullptr;
    this->coeff_b_x = nullptr;
    this->coeff_a_z = nullptr;
    this->coeff_b_z = nullptr;
    this->aux_1_x_up = nullptr;
    this->aux_1_x_down = nullptr;
    this->aux_1_z_up = nullptr;
    this->aux_1_z_down = nullptr;
    this->aux_2_x_up = nullptr;
    this->aux_2_x_down = nullptr;
    this->aux_2_z_up = nullptr;
    this->aux_2_z_down = nullptr;
    this->mpParameters = nullptr;
    this->mpGridBox = nullptr;
    this->max_vel = 0;
}

void CPMLBoundaryManager::AcquireConfiguration() {
    this->mUseTopLayer = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_USE_TOP_LAYER, this->mUseTopLayer);
    if (this->mUseTopLayer) {
        std::cout
                << "Using top boundary layer for forward modelling. To disable it set <boundary-manager.use-top-layer=false>"
                << std::endl;
    } else {
        std::cout
                << "Not using top boundary layer for forward modelling. To enable it set <boundary-manager.use-top-layer=true>"
                << std::endl;
    }
    if (this->mpConfigurationMap->Contains(OP_K_PROPRIETIES, OP_K_REFLECT_COEFFICIENT)) {
        std::cout << "Parsing user defined reflect coefficient" << std::endl;
        this->mReflectCoefficient = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_REFLECT_COEFFICIENT,
                                                                       this->mReflectCoefficient);
    }
    if (this->mpConfigurationMap->Contains(OP_K_PROPRIETIES, OP_K_SHIFT_RATIO)) {
        std::cout << "Parsing user defined shift ratio" << std::endl;
        this->mShiftRatio = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_SHIFT_RATIO, this->mShiftRatio);
    }
    if (this->mpConfigurationMap->Contains(OP_K_PROPRIETIES, OP_K_RELAX_COEFFICIENT)) {
        std::cout << "Parsing user defined relax coefficient" << std::endl;
        this->mRelaxCoefficient = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_RELAX_COEFFICIENT,
                                                                     this->mRelaxCoefficient);
    }

    this->mpExtension = new HomogenousExtension(this->mUseTopLayer);
    this->mpExtension->SetHalfLength(this->mpParameters->GetHalfLength());
    this->mpExtension->SetBoundaryLength(this->mpParameters->GetBoundaryLength());

}

void CPMLBoundaryManager::InitializeVariables() {
    int nx = this->mpGridBox->GetLogicalGridSize(X_AXIS);
    int nz = this->mpGridBox->GetLogicalGridSize(Z_AXIS);

    int actual_nx = this->mpGridBox->GetActualGridSize(X_AXIS);
    int actual_nz = this->mpGridBox->GetActualGridSize(Z_AXIS);

    int wnx = this->mpGridBox->GetLogicalWindowSize(X_AXIS);
    int wny = this->mpGridBox->GetLogicalWindowSize(Y_AXIS);
    int wnz = this->mpGridBox->GetLogicalWindowSize(Z_AXIS);

    float dt = this->mpGridBox->GetDT();
    int bound_length = this->mpParameters->GetBoundaryLength();
    int half_length = this->mpParameters->GetHalfLength();
    float *host_velocity_base = this->mpGridBox->Get(PARM | GB_VEL)->GetHostPointer();
    float max_velocity = 0;
    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i < nx; ++i) {
            int offset = i + actual_nx * j;
            float velocity_real = host_velocity_base[offset];
            if (velocity_real > max_velocity) {
                max_velocity = velocity_real;
            }
        }
    }

    max_velocity = max_velocity / (dt * dt);
    this->max_vel = sqrtf(max_velocity);

    this->coeff_a_x = new FrameBuffer<float>(bound_length);
    this->coeff_b_x = new FrameBuffer<float>(bound_length);
    this->coeff_a_z = new FrameBuffer<float>(bound_length);
    this->coeff_b_z = new FrameBuffer<float>(bound_length);

    int width = bound_length + (2 * this->mpParameters->GetHalfLength());
    int x_size = width * wny * wnz;
    int z_size = width * wnx * wny;

    this->aux_1_x_up = new FrameBuffer<float>(x_size);
    this->aux_1_x_down = new FrameBuffer<float>(x_size);
    this->aux_2_x_up = new FrameBuffer<float>(x_size);
    this->aux_2_x_down = new FrameBuffer<float>(x_size);

    Device::MemSet(aux_1_x_up->GetNativePointer(), 0.0, sizeof(float) * x_size);
    Device::MemSet(aux_1_x_down->GetNativePointer(), 0.0, sizeof(float) * x_size);
    Device::MemSet(aux_2_x_up->GetNativePointer(), 0.0, sizeof(float) * x_size);
    Device::MemSet(aux_2_x_down->GetNativePointer(), 0.0, sizeof(float) * x_size);

    this->aux_1_z_up = new FrameBuffer<float>(z_size);
    this->aux_1_z_down = new FrameBuffer<float>(z_size);
    this->aux_2_z_up = new FrameBuffer<float>(z_size);
    this->aux_2_z_down = new FrameBuffer<float>(z_size);

    Device::MemSet(aux_1_z_up->GetNativePointer(), 0.0, sizeof(float) * z_size);
    Device::MemSet(aux_1_z_down->GetNativePointer(), 0.0, sizeof(float) * z_size);
    Device::MemSet(aux_2_z_up->GetNativePointer(), 0.0, sizeof(float) * z_size);
    Device::MemSet(aux_2_z_down->GetNativePointer(), 0.0, sizeof(float) * z_size);

    FillCPMLCoefficients<1>();
    FillCPMLCoefficients<2>();
}

void CPMLBoundaryManager::ResetVariables() {
    int wnx = this->mpGridBox->GetLogicalWindowSize(X_AXIS);
    int wnz = this->mpGridBox->GetLogicalWindowSize(Z_AXIS);

    int width =
            this->mpParameters->GetBoundaryLength() + (2 * this->mpParameters->GetHalfLength());

    int x_size = width * wnz;
    int z_size = width * wnx;

    Device::MemSet(aux_1_x_up->GetNativePointer(), 0.0, sizeof(float) * x_size);
    Device::MemSet(aux_1_x_down->GetNativePointer(), 0.0, sizeof(float) * x_size);
    Device::MemSet(aux_2_x_up->GetNativePointer(), 0.0, sizeof(float) * x_size);
    Device::MemSet(aux_2_x_down->GetNativePointer(), 0.0, sizeof(float) * x_size);

    Device::MemSet(aux_1_z_up->GetNativePointer(), 0.0, sizeof(float) * z_size);
    Device::MemSet(aux_1_z_down->GetNativePointer(), 0.0, sizeof(float) * z_size);
    Device::MemSet(aux_2_z_up->GetNativePointer(), 0.0, sizeof(float) * z_size);
    Device::MemSet(aux_2_z_down->GetNativePointer(), 0.0, sizeof(float) * z_size);
}

void CPMLBoundaryManager::ExtendModel() {
    this->mpExtension->ExtendProperty();
    this->InitializeVariables();
}

void CPMLBoundaryManager::ReExtendModel() {
    this->mpExtension->ReExtendProperty();
    this->ResetVariables();
}

CPMLBoundaryManager::~CPMLBoundaryManager() {
    delete this->mpExtension;

    delete (coeff_a_x);
    delete (coeff_b_x);

    delete (coeff_a_z);
    delete (coeff_b_z);

    delete (aux_1_x_up);
    delete (aux_1_x_down);

    delete (aux_1_z_up);
    delete (aux_1_z_down);

    delete (aux_2_x_up);
    delete (aux_2_x_down);

    delete (aux_2_z_up);
    delete (aux_2_z_down);
}

void CPMLBoundaryManager::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CPMLBoundaryManager::SetGridBox(GridBox *apGridBox) {
    this->mpGridBox = apGridBox;
    if (this->mpGridBox == nullptr) {
        std::cerr << "No GridBox provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    /* Does not support 3D. */
    if (this->mpGridBox->GetActualWindowSize(Y_AXIS) > 1) {
        throw exceptions::NotImplementedException();
    }

    this->mpExtension->SetGridBox(this->mpGridBox);
    this->mpExtension->SetProperty(this->mpGridBox->Get(PARM | GB_VEL)->GetNativePointer(),
                                   this->mpGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer());
}

void CPMLBoundaryManager::ApplyBoundary(uint kernel_id) {
    if (kernel_id == 0) {
        switch (this->mpParameters->GetHalfLength()) {
            case O_2:
                ApplyAllCPML<O_2>();
                break;
            case O_4:
                ApplyAllCPML<O_4>();
                break;
            case O_8:
                ApplyAllCPML<O_8>();
                break;
            case O_12:
                ApplyAllCPML<O_12>();
                break;
            case O_16:
                ApplyAllCPML<O_16>();
                break;
        }
    }
}

void CPMLBoundaryManager::AdjustModelForBackward() {
    this->mpExtension->AdjustPropertyForBackward();
    this->ResetVariables();
}
