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
// Created by ahmed-ayyad on 29/11/2020.
//

#include <operations/components/independents/concrete/migration-accommodators/CrossCorrelationKernel.hpp>

#include <operations/utils/sampling/Sampler.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>

#define EPSILON 1e-20

using namespace std;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;
using namespace operations::utils::sampling;


CrossCorrelationKernel::CrossCorrelationKernel(operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mCompensationType = NO_COMPENSATION;
}

CrossCorrelationKernel::~CrossCorrelationKernel() {
    delete (this->mpShotCorrelation);
    delete (this->mpTotalCorrelation);
    delete (this->mpSourceIllumination);
    delete (this->mpReceiverIllumination);
    delete (this->mpTotalSourceIllumination);
    delete (this->mpTotalReceiverIllumination);
}

void CrossCorrelationKernel::AcquireConfiguration() {
    string compensation = "no";
    compensation = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_COMPENSATION, compensation);

    if (compensation.empty()) {
        cout << "No entry for migration-accommodator.compensation key : supported values [ "
                "no | source | receiver | combined ]"
             << std::endl;
        cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    } else if (compensation == OP_K_COMPENSATION_NONE) {
        this->SetCompensation(NO_COMPENSATION);
        cout << "No illumination compensation is requested" << std::endl;
    } else if (compensation == OP_K_COMPENSATION_SOURCE) {
        this->SetCompensation(SOURCE_COMPENSATION);
        cout << "Applying source illumination compensation" << std::endl;
    } else if (compensation == OP_K_COMPENSATION_RECEIVER) {
        this->SetCompensation(RECEIVER_COMPENSATION);
        cout << "Applying receiver illumination compensation" << std::endl;
    } else if (compensation == OP_K_COMPENSATION_COMBINED) {
        this->SetCompensation(COMBINED_COMPENSATION);
        cout << "Applying combined illumination compensation" << std::endl;
    } else {
        cout << "Invalid value for migration-accommodator.compensation key : supported values [ "
                OP_K_COMPENSATION_NONE " | "
                OP_K_COMPENSATION_SOURCE  " | "
                OP_K_COMPENSATION_RECEIVER  " | "
                OP_K_COMPENSATION_COMBINED " ]"
             << std::endl;
        cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CrossCorrelationKernel::Correlate(dataunits::DataUnit *apDataUnit) {
    auto grid_box = (GridBox *) apDataUnit;

    if (this->mpGridBox->GetLogicalGridSize(Y_AXIS) == 1) {
        switch (this->mCompensationType) {
            case NO_COMPENSATION:
                Correlation<true, NO_COMPENSATION>(grid_box);
                break;
            case COMBINED_COMPENSATION:
                Correlation<true, COMBINED_COMPENSATION>(grid_box);
                break;
        }
    } else {
        switch (this->mCompensationType) {
            case NO_COMPENSATION:
                Correlation<false, NO_COMPENSATION>(grid_box);
                break;
            case COMBINED_COMPENSATION:
                Correlation<false, COMBINED_COMPENSATION>(grid_box);
                break;
        }
    }
}

void CrossCorrelationKernel::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CrossCorrelationKernel::SetCompensation(COMPENSATION_TYPE aCOMPENSATION_TYPE) {
    mCompensationType = aCOMPENSATION_TYPE;
}

void CrossCorrelationKernel::SetGridBox(GridBox *apGridBox) {
    this->mpGridBox = apGridBox;
    if (this->mpGridBox == nullptr) {
        std::cerr << "No GridBox provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    /* Does not support 3D. */
    if (this->mpGridBox->GetActualWindowSize(Y_AXIS) > 1) {
        throw exceptions::NotImplementedException();
    }

    InitializeInternalElements();
}

void CrossCorrelationKernel::InitializeInternalElements() {
    // Grid.
    uint nx = this->mpGridBox->GetActualGridSize(X_AXIS);
    uint ny = this->mpGridBox->GetActualGridSize(Y_AXIS);
    uint nz = this->mpGridBox->GetActualGridSize(Z_AXIS);
    uint grid_size = nx * ny * nz;
    uint grid_bytes = grid_size * sizeof(float);

    // Window.
    uint wnx = this->mpGridBox->GetActualWindowSize(X_AXIS);
    uint wny = this->mpGridBox->GetActualWindowSize(Y_AXIS);
    uint wnz = this->mpGridBox->GetActualWindowSize(Z_AXIS);
    uint window_size = wnx * wny * wnz;
    uint window_bytes = window_size * sizeof(float);

    mpShotCorrelation = new FrameBuffer<float>();
    mpShotCorrelation->Allocate(window_size, mpParameters->GetHalfLength(), "shot_correlation");
    Device::MemSet(mpShotCorrelation->GetNativePointer(), 0, window_bytes);

    mpSourceIllumination = new FrameBuffer<float>();
    mpSourceIllumination->Allocate(window_size, mpParameters->GetHalfLength(), "source_illumination");
    Device::MemSet(mpSourceIllumination->GetNativePointer(), 0, window_bytes);

    mpReceiverIllumination = new FrameBuffer<float>();
    mpReceiverIllumination->Allocate(window_size, mpParameters->GetHalfLength(), "receiver_illumination");
    Device::MemSet(mpReceiverIllumination->GetNativePointer(), 0, window_bytes);

    mpTotalCorrelation = new FrameBuffer<float>();
    mpTotalCorrelation->Allocate(grid_size, mpParameters->GetHalfLength(), "stacked_shot_correlation");
    Device::MemSet(mpTotalCorrelation->GetNativePointer(), 0, grid_bytes);

    mpTotalSourceIllumination = new FrameBuffer<float>();
    mpTotalSourceIllumination->Allocate(grid_size, mpParameters->GetHalfLength(), "stacked_source_illumination");
    Device::MemSet(mpTotalSourceIllumination->GetNativePointer(), 0, grid_bytes);

    mpTotalReceiverIllumination = new FrameBuffer<float>();
    mpTotalReceiverIllumination->Allocate(grid_size, mpParameters->GetHalfLength(), "stacked_receiver_illumination");
    Device::MemSet(mpTotalReceiverIllumination->GetNativePointer(), 0, grid_bytes);
}

void CrossCorrelationKernel::ResetShotCorrelation() {
    uint window_bytes = sizeof(float) *
                        this->mpGridBox->GetActualWindowSize(X_AXIS) *
                        this->mpGridBox->GetActualWindowSize(Y_AXIS) *
                        this->mpGridBox->GetActualWindowSize(Z_AXIS);
    Device::MemSet(this->mpShotCorrelation->GetNativePointer(), 0, window_bytes);
    Device::MemSet(this->mpSourceIllumination->GetNativePointer(), 0, window_bytes);
    Device::MemSet(this->mpReceiverIllumination->GetNativePointer(), 0, window_bytes);
}

FrameBuffer<float> *CrossCorrelationKernel::GetShotCorrelation() {
    return this->mpShotCorrelation;
}

FrameBuffer<float> *CrossCorrelationKernel::GetStackedShotCorrelation() {
    return this->mpTotalCorrelation;
}

float *Unpad(float *apOriginalArray, uint nx, uint ny, uint nz,
             uint nx_original, uint ny_original, uint nz_original) {
    if (nx == nx_original && nz == nz_original && ny == ny_original) {
        return apOriginalArray;
    } else {
        auto copy_array = new float[ny_original * nz_original * nx_original];
        for (uint iy = 0; iy < ny_original; iy++) {
            for (uint iz = 0; iz < nz_original; iz++) {
                for (uint ix = 0; ix < nx_original; ix++) {
                    copy_array[iy * nz_original * nx_original + iz * nx_original + ix] =
                            apOriginalArray[iy * nz * nx + iz * nx + ix];
                }
            }
        }
        return copy_array;
    }
}

MigrationData *CrossCorrelationKernel::GetMigrationData() {
    vector<Result *> results;
    results.clear();
    uint lnx = this->mpGridBox->GetLogicalGridSize(X_AXIS);
    uint lny = this->mpGridBox->GetLogicalGridSize(Y_AXIS);
    uint lnz = this->mpGridBox->GetLogicalGridSize(Z_AXIS);
    uint nx = this->mpGridBox->GetActualGridSize(X_AXIS);
    uint ny = this->mpGridBox->GetActualGridSize(Y_AXIS);
    uint nz = this->mpGridBox->GetActualGridSize(Z_AXIS);
    switch (mCompensationType) {
        case NO_COMPENSATION:
            results.push_back(new Result(Unpad(
                    this->mpTotalCorrelation->GetHostPointer(),
                    nx, ny, nz,
                    lnx, lny, lnz)));
            break;
        case COMBINED_COMPENSATION:
            results.push_back(new Result(Unpad(this->mpTotalCorrelation->GetHostPointer(),
                                               nx, ny, nz,
                                               lnx, lny, lnz)));
            results.push_back(new Result(Unpad(this->mpTotalSourceIllumination->GetHostPointer(),
                                               nx, ny, nz,
                                               lnx, lny, lnz)));
            results.push_back(new Result(Unpad(this->mpTotalReceiverIllumination->GetHostPointer(),
                                               nx, ny, nz,
                                               lnx, lny, lnz)));
            break;
        default:
            results.push_back(new Result(Unpad(this->mpTotalCorrelation->GetHostPointer(),
                                               nx, ny, nz,
                                               lnx, lny, lnz)));
            break;
    }
    for (auto &result : results) {
        float *input = result->GetData();
        auto *output = new float[mpGridBox->GetInitialGridSize(X_AXIS) *
                                 mpGridBox->GetInitialGridSize(Y_AXIS) *
                                 mpGridBox->GetInitialGridSize(Z_AXIS)];
        Sampler::Resize(input, output,
                        mpGridBox->GetActualGridSize(), mpGridBox->GetInitialGridSize(),
                        mpParameters);
        if (!(nx == lnx && nz == lnz && ny == lny)) {
            delete[] input;
        }
        result = new Result(output);
    }
    return new MigrationData(this->mpGridBox->GetInitialGridSize(X_AXIS),
                             this->mpGridBox->GetInitialGridSize(Y_AXIS),
                             this->mpGridBox->GetInitialGridSize(Z_AXIS),
                             this->mpGridBox->GetNT(),
                             this->mpGridBox->GetInitialCellDimensions(X_AXIS),
                             this->mpGridBox->GetInitialCellDimensions(Y_AXIS),
                             this->mpGridBox->GetInitialCellDimensions(Z_AXIS),
                             this->mpGridBox->GetDT(),
                             results);
}
