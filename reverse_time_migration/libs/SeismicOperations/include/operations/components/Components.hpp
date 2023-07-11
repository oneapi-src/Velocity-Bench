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
// Created by zeyad-osama on 22/12/2020.
//

#ifndef OPERATIONS_LIB_COMPONENTS_COMPONENTS_HPP
#define OPERATIONS_LIB_COMPONENTS_COMPONENTS_HPP

/// COMPUTATION KERNELS
#include <operations/components/independents/concrete/computation-kernels/isotropic/SecondOrderComputationKernel.hpp>
#include <operations/components/independents/concrete/computation-kernels/isotropic/StaggeredComputationKernel.hpp>

/// MIGRATION ACCOMMODATORS
#include <operations/components/independents/concrete/migration-accommodators/CrossCorrelationKernel.hpp>

/// BOUNDARIES COMPONENTS
#include <operations/components/independents/concrete/boundary-managers/NoBoundaryManager.hpp>
#include <operations/components/independents/concrete/boundary-managers/RandomBoundaryManager.hpp>
#include <operations/components/independents/concrete/boundary-managers/SpongeBoundaryManager.hpp>
#include <operations/components/independents/concrete/boundary-managers/CPMLBoundaryManager.hpp>
#include <operations/components/independents/concrete/boundary-managers/StaggeredCPMLBoundaryManager.hpp>

/// FORWARD COLLECTORS
#include <operations/components/independents/concrete/forward-collectors/ReversePropagation.hpp>
#include <operations/components/independents/concrete/forward-collectors/TwoPropagation.hpp>

/// TRACE MANAGERS
#include <operations/components/independents/concrete/trace-managers/BinaryTraceManager.hpp>
#include <operations/components/independents/concrete/trace-managers/SeismicTraceManager.hpp>

///  SOURCE INJECTORS
#include <operations/components/independents/concrete/source-injectors/RickerSourceInjector.hpp>

/// MEMORY HANDLERS
#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>

/// MODEL HANDLERS
#include <operations/components/independents/concrete/model-handlers/SyntheticModelHandler.hpp>
#include <operations/components/independents/concrete/model-handlers/SeismicModelHandler.hpp>

/// MODELLING CONFIGURATION PARSERS
#include <operations/components/independents/concrete/modelling-configuration-parsers/TextModellingConfigurationParser.hpp>

/// TRACE WRITERS
#include <operations/components/independents/concrete/trace-writers/BinaryTraceWriter.hpp>


#endif //OPERATIONS_LIB_COMPONENTS_COMPONENTS_HPP
