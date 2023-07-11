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
// Created by mirna-moawad on 11/21/19.
//

#include "operations/components/independents/concrete/boundary-managers/SpongeBoundaryManager.hpp"

#include "operations/components/independents/concrete/boundary-managers/extensions/HomogenousExtension.hpp"

#include <iostream>
#include <cmath>

using namespace std;
using namespace operations::components;
using namespace operations::components::addons;
using namespace operations::common;
using namespace operations::dataunits;

/// Based on
/// https://pubs.geoscienceworld.org/geophysics/article-abstract/50/4/705/71992/A-nonreflecting-boundary-condition-for-discrete?redirectedFrom=fulltext

SpongeBoundaryManager::SpongeBoundaryManager(operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mUseTopLayer = true;
    this->mpSpongeCoefficients = nullptr;
}

void SpongeBoundaryManager::AcquireConfiguration() {
    this->mUseTopLayer = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_USE_TOP_LAYER, this->mUseTopLayer);
    if (this->mUseTopLayer) {
        cout
                << "Using top boundary layer for forward modelling. To disable it set <boundary-manager.use-top-layer=false>"
                << std::endl;
    } else {
        cout
                << "Not using top boundary layer for forward modelling. To enable it set <boundary-manager.use-top-layer=true>"
                << std::endl;
    }
}

float SpongeBoundaryManager::Calculation(int index) {
    float value;
    uint bound_length = mpParameters->GetBoundaryLength();
    value = expf(-powf((0.1f / bound_length) * (bound_length - index), 2));
    return value;
}

void SpongeBoundaryManager::ApplyBoundary(uint kernel_id) {
    if (kernel_id == 0) {
        ApplyBoundaryOnField(this->mpGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer());
    }
}

void SpongeBoundaryManager::ExtendModel() {
    for (auto const &extension : this->mvExtensions) {
        extension->ExtendProperty();
    }
}

void SpongeBoundaryManager::ReExtendModel() {
    for (auto const &extension : this->mvExtensions) {
        extension->ReExtendProperty();
    }
}

SpongeBoundaryManager::~SpongeBoundaryManager() {
    for (auto const &extension : this->mvExtensions) {
        delete extension;
    }
    this->mvExtensions.clear();
    delete this->mpSpongeCoefficients;
}

void SpongeBoundaryManager::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void SpongeBoundaryManager::SetGridBox(GridBox *apGridBox) {
    this->mpGridBox = apGridBox;
    if (this->mpGridBox == nullptr) {
        std::cerr << "No GridBox provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    /* Does not support 3D. */
    if (this->mpGridBox->GetActualWindowSize(Y_AXIS) > 1) {
        throw exceptions::NotImplementedException();
    }

    InitializeExtensions();

    uint bound_length = this->mpParameters->GetBoundaryLength();

    auto temp_arr = new float[bound_length];

    for (int i = 0; i < bound_length; i++) {
        temp_arr[i] = Calculation(i);
    }

    mpSpongeCoefficients = new FrameBuffer<float>(bound_length);
    Device::MemCpy(mpSpongeCoefficients->GetNativePointer(), temp_arr, bound_length * sizeof(float));

    delete[] temp_arr;
}

void SpongeBoundaryManager::InitializeExtensions() {
    uint params_size = this->mpGridBox->GetParameters().size();
    for (int i = 0; i < params_size; ++i) {
        this->mvExtensions.push_back(new HomogenousExtension(mUseTopLayer));
    }

    for (auto const &extension : this->mvExtensions) {
        extension->SetHalfLength(this->mpParameters->GetHalfLength());
        extension->SetBoundaryLength(this->mpParameters->GetBoundaryLength());
    }

    uint index = 0;
    for (auto const &parameter :  this->mpGridBox->GetParameters()) {
        this->mvExtensions[index]->SetGridBox(this->mpGridBox);
        this->mvExtensions[index]->SetProperty(parameter.second->GetNativePointer(),
                                               this->mpGridBox->Get(WIND | parameter.first)->GetNativePointer());
        index++;
    }
}

void SpongeBoundaryManager::AdjustModelForBackward() {
    for (auto const &extension : this->mvExtensions) {
        extension->AdjustPropertyForBackward();
    }
}
