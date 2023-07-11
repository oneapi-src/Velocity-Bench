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

if (USE_DPC)
    set(STBX-SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/concrete/computation-parameters/oneapi/computation_parameters_generator.cpp
            ${STBX-SOURCES}
            )
elseif (USE_CUDA)
    set(STBX-SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/concrete/computation-parameters/cuda/computation_parameters_generator.cpp
            ${STBX-SOURCES}
            )
elseif (USE_HIP)
    set(STBX-SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/concrete/computation-parameters/hip/computation_parameters_generator.cpp
            ${STBX-SOURCES}
            )
endif ()


set(STBX-SOURCES

        ${CMAKE_CURRENT_SOURCE_DIR}/Generator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitive/CallbacksGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitive/ComponentsGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitive/ConfigurationsGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitive/ComputationParametersGetter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitive/ComponentsGenerator.cpp

        ${STBX-SOURCES}
        PARENT_SCOPE
        )
