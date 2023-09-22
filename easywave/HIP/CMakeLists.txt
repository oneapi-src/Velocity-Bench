###  
###  Modifications Copyright (C) 2023 Intel Corporation
###  
###  This Program is subject to the terms of the European Union Public License 1.2
###  
###  If a copy of the license was not distributed with this file, you can obtain one at 
###  https://joinup.ec.europa.eu/sites/default/files/custom-page/attachment/2020-03/EUPL-1.2%20EN.txt
###  
###  SPDX-License-Identifier: EUPL-1.2
###  
###  

cmake_minimum_required(VERSION 3.10)
project(easywave_hip) 
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -O3 -Wextra -DUSE_HIP -DENABLE_HIP_LOGGING -D__HIP_PLATFORM_AMD__ ")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


option(SHOW_GRID               "Show intermediate grid size during propagation" OFF)
option(ENABLE_GPU_TIMINGS      "Show GPU timings at end of execution"           OFF)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

find_package(HIP REQUIRED)

### ROCM RELATED FLAGS START HERE
set(ROCM_WL_CXX_FLAGS       "")
set(ROCM_DEF_CXX_FLAGS       "-O3 -D__HIP_PLATFORM_AMD__ ")        
set(ROCM_DEF_COMBINED_FLAGS "${ROCM_WL_CXX_FLAGS} ${ROCM_DEF_CXX_FLAGS}")
### END OF ROCM RELATED FLAGS

if(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "" AND NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(FATAL_ERROR "Both  CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS cannot be passed in together")
elseif("${CMAKE_CXX_FLAGS}" STREQUAL "" AND "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags")
    set(CMAKE_CXX_FLAGS "${ROCM_DEF_COMBINED_FLAGS}")
elseif(NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(STATUS "OVERRIDING GENERAL compilation flags")
    set(CMAKE_CXX_FLAGS "${OVERRIDE_GENERAL_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS ${ROCM_WL_CXX_FLAGS})
elseif(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "OVERRIDING GENERAL and WORKLOAD SPECIFIC compilation flags")
endif()

if(SHOW_GRID)
    message(STATUS "Showing grid size during propagation")
    add_compile_options(-DSHOW_GRID)
endif()

if(ENABLE_GPU_TIMINGS)
    message(STATUS "GPU Timings will be displayed")
    add_compile_options(-DENABLE_GPU_TIMINGS)
endif()

# Host source files for compilation
set(SOURCES
    src/EasyWaveGen.cpp
    src/ewCudaKernels.cpp
    src/ewGpuNode.cpp
    ${CMAKE_SOURCE_DIR}/../common/cOgrd.cpp
    ${CMAKE_SOURCE_DIR}/../common/cOkadaEarthquake.cpp
    ${CMAKE_SOURCE_DIR}/../common/cOkadaFault.cpp
    ${CMAKE_SOURCE_DIR}/../common/cSphere.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewGrid.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewOut2D.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewParam.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewPOIs.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewReset.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewSource.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewStep.cpp
    ${CMAKE_SOURCE_DIR}/../common/okada.cpp
    ${CMAKE_SOURCE_DIR}/../common/utilits.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/FileHandler.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/Timer.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/Utilities.cpp
)
include_directories(${CMAKE_SOURCE_DIR}/../common ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/../../infrastructure)
link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
