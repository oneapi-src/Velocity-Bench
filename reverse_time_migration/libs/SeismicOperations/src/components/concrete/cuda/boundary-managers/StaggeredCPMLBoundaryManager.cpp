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
// Created by amr on 03/01/2021.
//
#include <operations/components/independents/concrete/boundary-managers/StaggeredCPMLBoundaryManager.hpp>

#include <operations/exceptions/NotImplementedException.h>

using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;


void StaggeredCPMLBoundaryManager::ApplyBoundary(uint kernel_id) {
    throw exceptions::NotImplementedException();
}

void StaggeredCPMLBoundaryManager::FillCPMLCoefficients(
        float *coeff_a, float *coeff_b, int boundary_length, float dh, float dt,
        float max_vel, float shift_ratio, float reflect_coeff, float relax_cp) {
    /// @todo
    /// {
    throw exceptions::NotImplementedException();
    /// }
}

// this function used to reset the auxiliary variables to zero
void StaggeredCPMLBoundaryManager::ZeroAuxiliaryVariables() {
    /// @todo
    /// {
    throw exceptions::NotImplementedException();
    /// }
}
