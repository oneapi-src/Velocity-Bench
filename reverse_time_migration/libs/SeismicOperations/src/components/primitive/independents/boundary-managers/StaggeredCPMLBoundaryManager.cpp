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

#include <operations/components/independents/concrete/boundary-managers/StaggeredCPMLBoundaryManager.hpp>

#include <operations/components/independents/concrete/boundary-managers/extensions/HomogenousExtension.hpp>

#include <iostream>
#include <algorithm>
#include <cmath>

#ifndef PWR2
#define PWR2(EXP) ((EXP) * (EXP))
#endif

#define fma(a, b, c) (((a) * (b)) + (c)

using namespace operations::components;
using namespace operations::components::addons;
using namespace operations::common;
using namespace operations::dataunits;

StaggeredCPMLBoundaryManager::StaggeredCPMLBoundaryManager(
        operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mReflectCoefficient = 0.1;
    this->mShiftRatio = 0.1;
    this->mRelaxCoefficient = 1;
    this->mUseTopLayer = true;
    this->mMaxVelocity = 0;
}

void StaggeredCPMLBoundaryManager::AcquireConfiguration() {
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
}

void StaggeredCPMLBoundaryManager::ExtendModel() {
    for (auto const &extension : this->mvExtensions) {
        extension->ExtendProperty();
    }
}

void StaggeredCPMLBoundaryManager::ReExtendModel() {
    for (auto const &extension : this->mvExtensions) {
        extension->ReExtendProperty();
    }
    ZeroAuxiliaryVariables();
}

StaggeredCPMLBoundaryManager::~StaggeredCPMLBoundaryManager() {
    for (auto const &extension : this->mvExtensions) {
        delete extension;
    }
    this->mvExtensions.clear();
}

void StaggeredCPMLBoundaryManager::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void StaggeredCPMLBoundaryManager::SetGridBox(GridBox *apGridBox) {
    this->mpGridBox = apGridBox;
    if (this->mpGridBox == nullptr) {
        std::cerr << "No GridBox provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
    InitializeExtensions();

    int nx = this->mpGridBox->GetLogicalGridSize(X_AXIS);
    int ny = this->mpGridBox->GetLogicalGridSize(Y_AXIS);
    int nz = this->mpGridBox->GetLogicalGridSize(Z_AXIS);

    int actual_nx = this->mpGridBox->GetActualGridSize(X_AXIS);
    int actual_ny = this->mpGridBox->GetActualGridSize(Y_AXIS);
    int actual_nz = this->mpGridBox->GetActualGridSize(Z_AXIS);

    int wnx = this->mpGridBox->GetLogicalWindowSize(X_AXIS);
    int wny = this->mpGridBox->GetLogicalWindowSize(Y_AXIS);
    int wnz = this->mpGridBox->GetLogicalWindowSize(Z_AXIS);

    int b_l = mpParameters->GetBoundaryLength();
    HALF_LENGTH h_l = mpParameters->GetHalfLength();

    // get the size of grid
    int grid_total_size = wnx * wnz * wny;
    // the size of the boundary in x without half_length is x=b_l and z=nz-2*h_l
    int bound_size_x = b_l * (wnz - 2 * h_l);
    // the size of the boundary in z without half_length is x=nx-2*h_l and z=b_l
    int bound_size_z = b_l * (wnx - 2 * h_l);

    int bound_size_y = 0;
    bool is_2d;
    int y_start;
    int y_end;

    if (wny == 1) {
        is_2d = 1;
        y_start = 0;
        y_end = 1;
    } else {
        y_start = h_l;
        y_end = wny - h_l;
        is_2d = 0;
        // size of boundary in y without half_length x=nx-2*h_l z=nz-2*h_l y= b_l
        bound_size_y = b_l * (wnx - 2 * h_l) * (wnz - 2 * h_l);

        // if 3d multiply by y size without half_length which is ny-2*h_l
        bound_size_x = bound_size_x * (wny - 2 * h_l);
        bound_size_z = bound_size_z * (wny - 2 * h_l);
    }

    // allocate the small arrays for coefficients
    small_a_x = new FrameBuffer<float>(b_l);
    small_a_z = new FrameBuffer<float>(b_l);
    small_b_x = new FrameBuffer<float>(b_l);
    small_b_z = new FrameBuffer<float>(b_l);

    // allocate the auxiliary variables for the boundary length for velocity in x
    // direction
    auxiliary_vel_x_left = new FrameBuffer<float>(bound_size_x);
    auxiliary_vel_x_right = new FrameBuffer<float>(bound_size_x);

    // allocate the auxiliary variables for the boundary length for velocity in z
    // direction
    auxiliary_vel_z_up = new FrameBuffer<float>(bound_size_z);
    auxiliary_vel_z_down = new FrameBuffer<float>(bound_size_z);

    // allocate the auxiliary variables for the boundary length for pressure in x
    // direction
    auxiliary_ptr_x_left = new FrameBuffer<float>(bound_size_x);
    auxiliary_ptr_x_right = new FrameBuffer<float>(bound_size_x);

    // allocate the auxiliary variables for the boundary length for pressure in z
    // direction
    auxiliary_ptr_z_up = new FrameBuffer<float>(bound_size_z);
    auxiliary_ptr_z_down = new FrameBuffer<float>(bound_size_z);

    // get the maximum velocity
    float *velocity_base = this->mpGridBox->Get(PARM | GB_VEL)->GetHostPointer();
    for (int k = 0; k < ny; ++k) {
        for (int j = 0; j < nz; ++j) {
            for (int i = 0; i < nx; ++i) {
                int offset = i + actual_nx * j + k * actual_nx * actual_nz;
                float velocity_real = velocity_base[offset];
                if (velocity_real > this->mMaxVelocity) {
                    this->mMaxVelocity = velocity_real;
                }
            }
        }
    }

    /// Put values for the small arrays

    StaggeredCPMLBoundaryManager::FillCPMLCoefficients(
            small_a_x->GetNativePointer(), small_b_x->GetNativePointer(), b_l,
            this->mpGridBox->GetCellDimensions(X_AXIS), this->mpGridBox->GetDT(),
            this->mMaxVelocity, this->mShiftRatio, this->mReflectCoefficient, this->mRelaxCoefficient);

    StaggeredCPMLBoundaryManager::FillCPMLCoefficients(
            small_a_z->GetNativePointer(), small_b_z->GetNativePointer(), b_l,
            this->mpGridBox->GetCellDimensions(Z_AXIS), this->mpGridBox->GetDT(),
            this->mMaxVelocity, this->mShiftRatio, this->mReflectCoefficient, this->mRelaxCoefficient);
}


void StaggeredCPMLBoundaryManager::InitializeExtensions() {
    uint params_size = this->mpGridBox->GetParameters().size();
    for (int i = 0; i < params_size; ++i) {
        this->mvExtensions.push_back(new HomogenousExtension(this->mUseTopLayer));
    }

    for (auto const &extension : this->mvExtensions) {
        extension->SetHalfLength(this->mpParameters->GetHalfLength());
        extension->SetBoundaryLength(this->mpParameters->GetBoundaryLength());
    }

    uint index = 0;
    for (auto const &parameter : this->mpGridBox->GetParameters()) {
        this->mvExtensions[index]->SetGridBox(this->mpGridBox);
        this->mvExtensions[index]->SetProperty(parameter.second->GetNativePointer(),
                                               this->mpGridBox->Get(WIND | parameter.first)->GetNativePointer());
        index++;
    }
}

void StaggeredCPMLBoundaryManager::AdjustModelForBackward() {
    for (auto const &extension : this->mvExtensions) {
        extension->AdjustPropertyForBackward();
    }
    ZeroAuxiliaryVariables();
}
