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

#include "operations/components/independents/concrete/boundary-managers/RandomBoundaryManager.hpp"

#include "operations/components/independents/concrete/boundary-managers/extensions/MinExtension.hpp"
#include "operations/components/independents/concrete/boundary-managers/extensions/RandomExtension.hpp"

#include <cstdlib>
#include <ctime>

using namespace operations::components;
using namespace operations::components::addons;
using namespace operations::common;
using namespace operations::dataunits;

// If the constructor is not given any parameters.
RandomBoundaryManager::RandomBoundaryManager(operations::configuration::ConfigurationMap *apConfigurationMap) {
    srand(time(NULL));
    this->mpConfigurationMap = apConfigurationMap;
}

void RandomBoundaryManager::AcquireConfiguration() {}

RandomBoundaryManager::~RandomBoundaryManager() {
    for (auto const &extension : this->mvExtensions) {
        delete extension;
    }
    this->mvExtensions.clear();
}

void RandomBoundaryManager::ExtendModel() {
    for (auto const &extension : this->mvExtensions) {
        extension->ExtendProperty();
    }
}

void RandomBoundaryManager::ReExtendModel() {
    for (auto const &extension : this->mvExtensions) {
        extension->ExtendProperty();
        extension->ReExtendProperty();
    }
}

void RandomBoundaryManager::ApplyBoundary(uint kernel_id) {
    // Do nothing for random boundaries.
}

void RandomBoundaryManager::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void RandomBoundaryManager::SetGridBox(GridBox *apGridBox) {
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
}

void RandomBoundaryManager::InitializeExtensions() {
    this->mvExtensions.push_back(new RandomExtension());

    uint params_size = this->mpGridBox->GetParameters().size();
    for (int i = 0; i < params_size - 1; ++i) {
        this->mvExtensions.push_back(new MinExtension());
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

void RandomBoundaryManager::AdjustModelForBackward() {
    for (auto const &extension : this->mvExtensions) {
        extension->AdjustPropertyForBackward();
    }
}
