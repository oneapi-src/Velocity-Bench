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
// Created by zeyad-osama on 18/08/2020.
//

#include <operations/components/independents/concrete/forward-collectors/ReversePropagation.hpp>

#include <operations/components/independents/concrete/forward-collectors/boundary-saver/boundary_saver.h>

using namespace operations::components;
using namespace operations::components::helpers;
using namespace operations::helpers;
using namespace operations::common;
using namespace operations::dataunits;


ReversePropagation::ReversePropagation(operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mpInternalGridBox = new GridBox();
    this->mpComputationKernel = nullptr;

    this->mInjectionEnabled = false;

    this->mBoundariesSize = 0;
    this->mTimeStep = 0;
    this->mpBackupBoundaries = nullptr;
}

ReversePropagation::~ReversePropagation() {
    this->mpWaveFieldsMemoryHandler->FreeWaveFields(this->mpInternalGridBox);
    delete this->mpInternalGridBox;
    delete this->mpComputationKernel;
}

void ReversePropagation::AcquireConfiguration() {
    this->mInjectionEnabled = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_BOUNDARY_SAVING,
                                                                 this->mInjectionEnabled);
    auto computation_kernel = (ComputationKernel *) (this->mpComponentsMap->Get(COMPUTATION_KERNEL));

    this->mpComputationKernel = (ComputationKernel *) computation_kernel->Clone();
    this->mpComputationKernel->SetDependentComponents(this->GetDependentComponentsMap());
    this->mpComputationKernel->SetAdjoint(true);
    this->mpComputationKernel->SetComputationParameters(this->mpParameters);
    this->mpComputationKernel->SetGridBox(this->mpInternalGridBox);
    this->mpComputationKernel->SetDependentComponents(this->GetDependentComponentsMap());
}

void ReversePropagation::FetchForward() {
    this->mpComputationKernel->Step();

    if (this->mInjectionEnabled) {
        this->mTimeStep--;
        restore_boundaries(this->mpMainGridBox,
                           this->mpInternalGridBox,
                           this->mpParameters,
                           this->mpBackupBoundaries,
                           this->mTimeStep,
                           this->mBoundariesSize);
    }
}

void ReversePropagation::ResetGrid(bool is_forward_run) {
    uint wnx = this->mpMainGridBox->GetActualWindowSize(X_AXIS);
    uint wny = this->mpMainGridBox->GetActualWindowSize(Y_AXIS);
    uint wnz = this->mpMainGridBox->GetActualWindowSize(Z_AXIS);
    uint const window_size = wnx * wny * wnz;

    if (!is_forward_run) {
        this->mpMainGridBox->CloneMetaData(this->mpInternalGridBox);
        this->mpMainGridBox->CloneParameters(this->mpInternalGridBox);
        if (this->mpInternalGridBox->GetWaveFields().empty()) {
            this->mpWaveFieldsMemoryHandler->CloneWaveFields(this->mpMainGridBox, this->mpInternalGridBox);
        } else {
            this->mpWaveFieldsMemoryHandler->CopyWaveFields(this->mpMainGridBox, this->mpInternalGridBox);
        }

        /*
         * Swapping
         */

        if (this->mpParameters->GetApproximation() == ISOTROPIC) {
            if (this->mpParameters->GetEquationOrder() == FIRST) {
                // Swap next and current to reverse time.
                this->mpInternalGridBox->Swap(WAVE | GB_PRSS | NEXT | DIR_Z,
                                              WAVE | GB_PRSS | CURR | DIR_Z);
            } else if (this->mpParameters->GetEquationOrder() == SECOND) {
                // Swap previous and current to reverse time.
                this->mpInternalGridBox->Swap(WAVE | GB_PRSS | PREV | DIR_Z,
                                              WAVE | GB_PRSS | CURR | DIR_Z);
                // Only use two pointers, prev is same as next.
                this->mpInternalGridBox->Set(WAVE | GB_PRSS | NEXT | DIR_Z,
                                             this->mpInternalGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z));
            }
        } else if (this->mpParameters->GetApproximation() == VTI ||
                   this->mpParameters->GetApproximation() == TTI) {
            // Swap vertical previous and current to reverse time.
            this->mpInternalGridBox->Swap(WAVE | GB_PRSS | PREV | DIR_X,
                                          WAVE | GB_PRSS | CURR | DIR_X);

            // Swap horizontal  previous and current to reverse time.
            this->mpInternalGridBox->Swap(WAVE | GB_PRSS | PREV | DIR_Z, WAVE | GB_PRSS | CURR | DIR_Z);

            /// Using only two pointers
            // Only use two pointers, prev is same as next for vertical.
            this->mpInternalGridBox->Set(WAVE | GB_PRSS | NEXT | DIR_X,
                                         this->mpInternalGridBox->Get(WAVE | GB_PRSS | PREV | DIR_X));
            // Only use two pointers, prev is same as next for horizontal .
            this->mpInternalGridBox->Set(WAVE | GB_PRSS | NEXT | DIR_Z,
                                         this->mpInternalGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z));
        }
    } else {
        if (this->mInjectionEnabled) {
            this->Inject();
        }
    }

    for (auto const &wave_field : this->mpMainGridBox->GetWaveFields()) {
        Device::MemSet(wave_field.second->GetNativePointer(), 0.0f, window_size * sizeof(float));
    }
}

void ReversePropagation::SaveForward() {
    if (this->mInjectionEnabled) {
        save_boundaries(this->mpMainGridBox,
                        this->mpParameters,
                        this->mpBackupBoundaries,
                        this->mTimeStep,
                        this->mBoundariesSize);
        this->mTimeStep++;
    }
}

void ReversePropagation::Inject() {
    if (this->mpBackupBoundaries != nullptr) {
        mem_free(this->mpBackupBoundaries);
    }
    this->mTimeStep = 0;
    uint half_length = this->mpParameters->GetHalfLength();
    uint bound_length = this->mpParameters->GetBoundaryLength();
    uint nxi = this->mpMainGridBox->GetLogicalWindowSize(X_AXIS) - 2 * (half_length + bound_length);
    uint nzi = this->mpMainGridBox->GetLogicalWindowSize(Z_AXIS) - 2 * (half_length + bound_length);
    uint nyi = this->mpMainGridBox->GetLogicalWindowSize(Y_AXIS);
    if (nyi != 1) {
        nyi = nyi - 2 * (half_length + bound_length);
    }

    this->mBoundariesSize =
            nxi * nyi * half_length * 2 + nzi * nyi * half_length * 2;
    if (nyi != 1) {
        this->mBoundariesSize += nxi * nzi * half_length * 2;
    }
    this->mpBackupBoundaries =
            (float *) mem_allocate(
                    sizeof(float), this->mBoundariesSize * (this->mpMainGridBox->GetNT() + 1),
                    "boundary memory");
}

void ReversePropagation::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void ReversePropagation::SetGridBox(GridBox *apGridBox) {
    this->mpMainGridBox = apGridBox;
    if (this->mpMainGridBox == nullptr) {
        std::cout << "Not a compatible GridBox... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    /* Does not support 3D. */
    if (this->mpMainGridBox->GetActualWindowSize(Y_AXIS) > 1) {
        throw exceptions::NotImplementedException();
    }
}

void ReversePropagation::SetDependentComponents(
        ComponentsMap<DependentComponent> *apDependentComponentsMap) {
    HasDependents::SetDependentComponents(apDependentComponentsMap);

    this->mpWaveFieldsMemoryHandler =
            (WaveFieldsMemoryHandler *)
                    this->GetDependentComponentsMap()->Get(MEMORY_HANDLER);
    if (this->mpWaveFieldsMemoryHandler == nullptr) {
        std::cerr << "No Wave Fields Memory Handler provided... "
                  << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

GridBox *ReversePropagation::GetForwardGrid() {
    return this->mpInternalGridBox;
}
