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

#include <operations/components/independents/concrete/source-injectors/RickerSourceInjector.hpp>
#include <iostream>
#include <cmath>

using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;

RickerSourceInjector::RickerSourceInjector(operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
}

RickerSourceInjector::~RickerSourceInjector() = default;

void RickerSourceInjector::AcquireConfiguration() {}

void RickerSourceInjector::ApplyIsotropicField() {
    /// @todo To be implemented.

    uint location = this->GetInjectionLocation();
    uint isotropic_radius = this->mpParameters->GetIsotropicRadius();

    // Loop on circular field
    //      1. epsilon[location] = 0.0f
    //      2. delta[location] = 0.0f
}

void RickerSourceInjector::RevertIsotropicField() {
    /// @todo To be implemented.
}

uint RickerSourceInjector::GetCutOffTimeStep() {
    float dt = this->mpGridBox->GetDT();
    float freq = this->mpParameters->GetSourceFrequency();
    return (2.0 / freq) / dt;
}

void RickerSourceInjector::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void RickerSourceInjector::SetGridBox(GridBox *apGridBox) {
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

void RickerSourceInjector::SetSourcePoint(Point3D *apSourcePoint) {
    this->mpSourcePoint = apSourcePoint;
}

uint RickerSourceInjector::GetInjectionLocation() {
    uint x = this->mpSourcePoint->x;
    uint y = this->mpSourcePoint->y;
    uint z = this->mpSourcePoint->z;

    uint wnx = this->mpGridBox->GetActualWindowSize(X_AXIS);
    uint wny = this->mpGridBox->GetActualWindowSize(Y_AXIS);
    uint wnz = this->mpGridBox->GetActualWindowSize(Z_AXIS);

    uint location = y * wnx * wnz + z * wnx + x;
    return location;
}