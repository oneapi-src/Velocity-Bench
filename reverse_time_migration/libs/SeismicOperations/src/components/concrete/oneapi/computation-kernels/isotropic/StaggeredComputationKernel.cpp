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

#include <operations/components/independents/concrete/computation-kernels/isotropic/StaggeredComputationKernel.hpp>

#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>
#include <operations/exceptions/NotImplementedException.h>

using namespace std;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;

template void StaggeredComputationKernel::Compute<true, O_2>();

template void StaggeredComputationKernel::Compute<true, O_4>();

template void StaggeredComputationKernel::Compute<true, O_8>();

template void StaggeredComputationKernel::Compute<true, O_12>();

template void StaggeredComputationKernel::Compute<true, O_16>();

template void StaggeredComputationKernel::Compute<false, O_2>();

template void StaggeredComputationKernel::Compute<false, O_4>();

template void StaggeredComputationKernel::Compute<false, O_8>();

template void StaggeredComputationKernel::Compute<false, O_12>();

template void StaggeredComputationKernel::Compute<false, O_16>();

template void StaggeredComputationKernel::Compute<true, O_2>();

template void StaggeredComputationKernel::Compute<true, O_4>();

template void StaggeredComputationKernel::Compute<true, O_8>();

template void StaggeredComputationKernel::Compute<true, O_12>();

template void StaggeredComputationKernel::Compute<true, O_16>();

template void StaggeredComputationKernel::Compute<false, O_2>();

template void StaggeredComputationKernel::Compute<false, O_4>();

template void StaggeredComputationKernel::Compute<false, O_8>();

template void StaggeredComputationKernel::Compute<false, O_12>();

template void StaggeredComputationKernel::Compute<false, O_16>();


template<bool IS_FORWARD_, HALF_LENGTH HALF_LENGTH_>
void StaggeredComputationKernel::Compute() {
    throw exceptions::NotImplementedException();
}
