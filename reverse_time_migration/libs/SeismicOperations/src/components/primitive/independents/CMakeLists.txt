### 
### Modifications Copyright (C) 2023 Intel Corporation
### 
### This Program is subject to the terms of the GNU Lesser General Public License v3.0 or later
### 
### If a copy of the license was not distributed with this file, you can obtain one at 
### https://www.gnu.org/licenses/lgpl-3.0-standalone.html
### 
### SPDX-License-Identifier: LGPL-3.0-or-later
### 
### 

set(OPERATIONS-SOURCES

        # COMPUTATION KERNELS
        ${CMAKE_CURRENT_SOURCE_DIR}/computation-kernels/isotropic/SecondOrderComputationKernel.cpp
        #${CMAKE_CURRENT_SOURCE_DIR}/computation-kernels/isotropic/StaggeredComputationKernel.cpp

        # MIGRATION ACCOMMODATORS
        ${CMAKE_CURRENT_SOURCE_DIR}/migration-accommodators/CrossCorrelationKernel.cpp

        # BOUNDARIES COMPONENTS
        ${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/extensions/Extension.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/NoBoundaryManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/RandomBoundaryManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/SpongeBoundaryManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/CPMLBoundaryManager.cpp
        #${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/StaggeredCPMLBoundaryManager.cpp

        # FORWARD COLLECTORS
        ${CMAKE_CURRENT_SOURCE_DIR}/forward-collectors/ReversePropagation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/forward-collectors/TwoPropagation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/forward-collectors/boundary-saver/boundary_saver.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/forward-collectors/file-handler/file_handler.cpp

        # FORWARD COLLECTORS' HELPERS
        ${CMAKE_CURRENT_SOURCE_DIR}/forward-collectors/file-handler/file_handler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/forward-collectors/boundary-saver/boundary_saver.cpp

        # TRACE MANAGERS
        ${CMAKE_CURRENT_SOURCE_DIR}/trace-managers/BinaryTraceManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trace-managers/SeismicTraceManager.cpp

        # SOURCE INJECTORS
        ${CMAKE_CURRENT_SOURCE_DIR}/source-injectors/RickerSourceInjector.cpp

        # MODEL HANDLERS
        ${CMAKE_CURRENT_SOURCE_DIR}/model-handlers/SyntheticModelHandler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/model-handlers/SeismicModelHandler.cpp

        # TRACE WRITERS
        ${CMAKE_CURRENT_SOURCE_DIR}/trace-writers/BinaryTraceWriter.cpp

        # MODELLING CONFIGURATION PARSER
        ${CMAKE_CURRENT_SOURCE_DIR}/modelling-configuration-parsers/TextModellingConfigurationParser.cpp

        ${OPERATIONS-SOURCES}
        PARENT_SCOPE
        )
