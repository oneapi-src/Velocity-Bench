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
project(easyWave_sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)           # SYCL code requires this
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Enable modern C++ standards
set(CMAKE_CXX_EXTENSIONS OFF)        # Use -std, not -gnu

option(ENABLE_KERNEL_PROFILING "Build using kernel profiling"                   OFF)
option(GPU_AOT                 "Build AOT for Intel GPU"                        OFF)
option(USE_INTEL_CPU           "Build AOT for Intel CPU"                        OFF)
option(USE_NVIDIA_BACKEND      "Build for NVIDIA backend"                       OFF)
option(USE_AMDHIP_BACKEND      "Build for AMD HIP backend"                      OFF)
option(SHOW_GRID               "Show intermediate grid size during propagation" OFF)
option(ENABLE_GPU_TIMINGS      "Show GPU timings at end of execution"           OFF)

set(SOURCES
    src/EasyWave.cpp
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
    ${CMAKE_SOURCE_DIR}/../../infrastructure/SYCL.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/Timer.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/Utilities.cpp
)

include_directories(${CMAKE_SOURCE_DIR}/../common/ ${CMAKE_SOURCE_DIR}/src/ ${CMAKE_SOURCE_DIR}/../../infrastructure)

if(ENABLE_KERNEL_PROFILING)
    message(STATUS "Enabling kernel profiling")
    add_compile_options(-DENABLE_KERNEL_PROFILING)
endif()

if(SHOW_GRID)
    message(STATUS "Showing grid size during propagation")
    add_compile_options(-DSHOW_GRID)
endif()

if(ENABLE_GPU_TIMINGS)
    message(STATUS "GPU Timings will be displayed")
    add_compile_options(-DENABLE_GPU_TIMINGS)
endif()

# Use either default or user defined CXX flags
# -DCMAKE_CXX_FLAGS=" -blah -blah " overrides the default flags

set(USE_DEFAULT_FLAGS ON)
set(INTEL_GPU_CXX_FLAGS  " -O2 -fsycl -ffast-math ")
set(NVIDIA_GPU_CXX_FLAGS " -O3 -fsycl -ffast-math ")
set(AMD_GPU_CXX_FLAGS    " -O3 -fsycl -ffast-math ")

if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags for the application")
    string(APPEND CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}") # Default flags for NV backend
else()
    message(STATUS "OVERRIDING compilation flags")
    set(USE_DEFAULT_FLAGS OFF)
endif()

# JIT compilation 
if(GPU_AOT) 
    if( (${GPU_AOT} STREQUAL "pvc") OR (${GPU_AOT} STREQUAL "PVC") )
        message(STATUS "Enabling Intel GPU AOT compilation for ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_gen -Xs \"-device 0x0bd5 -revision_id 0x2f\" ") 
    else()
        message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
    endif()
elseif(USE_INTEL_CPU)
    message(STATUS "Compiling for Intel CPU")
    string(APPEND CMAKE_CXX_FLAGS "  -ffast-math -mprefer-vector-width=512 -mfma -fsycl-targets=spir64_x86_64 \"-device avx512\" ")
elseif(USE_NVIDIA_BACKEND)    
    message(STATUS "Enabling NVIDIA backend")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}") # Default flags for NV backend
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda ") # -O3 will be used, even though -O2 was set earlier 
    if(USE_SM)
        message(STATUS "Building for SM_${USE_SM} architecture")
        string(APPEND CMAKE_CXX_FLAGS " -Xsycl-target-backend --cuda-gpu-arch=sm_${USE_SM} ")
    endif()
elseif(USE_AMDHIP_BACKEND)
    message(STATUS "Enabling AMD HIP backend for ${USE_AMDHIP_BACKEND} AMD architecture")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}")
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${USE_AMDHIP_BACKEND}  ")
endif()

# Output the compiler flags that were constructed for visual inspection
message(STATUS "Compilation flags set to: ${CMAKE_CXX_FLAGS}")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} sycl stdc++fs)
