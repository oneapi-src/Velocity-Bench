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

#include <operations/exceptions/Exceptions.h>

using namespace operations::components;


template<int HALF_LENGTH_>
void CPMLBoundaryManager::ApplyAllCPML() {
    /// @todo
    /// {
    throw exceptions::NotImplementedException();
    /// }
}

// make them private to the class , extend to th 3d
template<int DIRECTION_>
void CPMLBoundaryManager::FillCPMLCoefficients() {
    /// @todo
    /// {
    throw exceptions::NotImplementedException();
    /// }
}

template<int DIRECTION_, bool OPPOSITE_, int HALF_LENGTH_>
void CPMLBoundaryManager::CalculateFirstAuxiliary() {
    /// @todo
    /// {
    throw exceptions::NotImplementedException();
    /// }
}

template<int DIRECTION_, bool OPPOSITE_, int HALF_LENGTH_>
void CPMLBoundaryManager::CalculateCPMLValue() {
    /// @todo
    /// {
    throw exceptions::NotImplementedException();
    /// }
}
