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
         ${CMAKE_CURRENT_SOURCE_DIR}/computation-kernels/isotropic/SecondOrderComputationKernel.cpp ### Using CUDA
        #${CMAKE_CURRENT_SOURCE_DIR}/computation-kernels/isotropic/StaggeredComputationKernel.cpp

        # MIGRATION ACCOMMODATORS
        ${CMAKE_CURRENT_SOURCE_DIR}/migration-accommodators/CrossCorrelationKernel.cpp ### Using CUDA

        # BOUNDARIES COMPONENTS
        ${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/extensions/ZeroExtension.cpp ### Using CUDA
        ${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/extensions/RandomExtension.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/extensions/HomogenousExtension.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/extensions/MinExtension.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/SpongeBoundaryManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/CPMLBoundaryManager.cpp
        #${CMAKE_CURRENT_SOURCE_DIR}/boundary-managers/StaggeredCPMLBoundaryManager.cpp

        # TRACE MANAGERS
        ${CMAKE_CURRENT_SOURCE_DIR}/trace-managers/SeismicTraceManager.cpp ### Using CUDA
        ${CMAKE_CURRENT_SOURCE_DIR}/trace-managers/BinaryTraceManager.cpp

        # SOURCE INJECTORS
        ${CMAKE_CURRENT_SOURCE_DIR}/source-injectors/RickerSourceInjector.cpp ### Using CUDA

        # MEMORY HANDLERS
        ${CMAKE_CURRENT_SOURCE_DIR}/memory-handlers/WaveFieldsMemoryHandler.cpp

        # MODEL HANDLERS
        ${CMAKE_CURRENT_SOURCE_DIR}/model-handlers/SyntheticModelHandler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/model-handlers/SeismicModelHandler.cpp  ### Using CUDA

        # FRAME BUFFER
        ${CMAKE_CURRENT_SOURCE_DIR}/data-units/FrameBuffer.cpp

        # TRACE WRITERS
        ${CMAKE_CURRENT_SOURCE_DIR}/trace-writers/BinaryTraceWriter.cpp

        ${OPERATIONS-SOURCES}
        PARENT_SCOPE
        )
