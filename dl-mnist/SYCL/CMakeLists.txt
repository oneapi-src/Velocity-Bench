 # Copyright (C) 2023 Intel Corporation
 # 
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom
 # the Software is furnished to do so, subject to the following conditions:
 # 
 # The above copyright notice and this permission notice shall be included
 # in all copies or substantial portions of the Software.
 # 
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 # OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 # OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 # ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 # OR OTHER DEALINGS IN THE SOFTWARE.
 # 
 # SPDX-License-Identifier: MIT
 #
 
cmake_minimum_required(VERSION 3.10)
project(dl-mnist-sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)           # SYCL code requires this
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Enable modern C++ standards
set(CMAKE_CXX_EXTENSIONS OFF)        # Use -std, not -gnu


option(GPU_AOT                 "Build AOT for Intel GPU"      OFF)
option(USE_NVIDIA_BACKEND      "Build for NVIDIA backend"     OFF)
option(USE_AMDHIP_BACKEND      "Build for AMD HIP backend"    OFF)
option(DEVICE_TIMER            "Build using Device Timer" OFF)

if(NOT DEFINED DNNLROOT AND DEFINED ENV{DNNLROOT})
    set(DNNLROOT "$ENV{DNNLROOT}" CACHE STRING "")
else()
    set(DNNLROOT "${PROJECT_SOURCE_DIR}/.." CACHE STRING "")
endif()



set(SOURCES 
    conv_layer.onednn.cpp
    tensor_mgr.onednn.cpp
    dl_network_mgr.onednn.cpp
    main.onednn.cpp
    ../common/mnist.cpp
    ../common/timing.cpp
    ../common/exec_policies.cpp
    ../common/utils.cpp
    ../common/workload_params.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/SYCL.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/CommandLineParser.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/TestBenchBase.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/Utilities.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../common
    ${CMAKE_SOURCE_DIR}/../../infrastructure
    ${DNNLROOT}/include
    
)

link_directories(${DNNLROOT}/lib)

message(STATUS "DNNLROOT set to: ${DNNLROOT}")

set(INTEL_GPU_CXX_FLAGS  " -O2 -fsycl -ffast-math")
set(NVIDIA_GPU_CXX_FLAGS " -O2 -fsycl -ffast-math")
set(AMD_GPU_CXX_FLAGS    " -O2 -fsycl -ffast-math")

set(USE_DEFAULT_FLAGS ON)
if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags")
else()
    message(STATUS "OVERRIDING DEFAULT compilation flags")
    set(USE_DEFAULT_FLAGS OFF)
endif()

add_compile_options(-DRUN_ON_GPU)
#add_compile_options(-DKERNEL_USE_PROFILE "0")
if(DEVICE_TIMER)
    message(STATUS "Enabling Device Timer")
    add_compile_options(-DDEVICE_TIMER)
endif()

# JIT compilation 
if(GPU_AOT) 
    message(STATUS "Enabling INTEL backend")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}") # Default flags for Intel backend
    endif()
    if( (${GPU_AOT} STREQUAL "pvc") OR (${GPU_AOT} STREQUAL "PVC") )
        message(STATUS "Enabling Intel GPU AOT compilation for ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_gen -Xs \"-device 0x0bd5 -revision_id 0x2f\" ") 
    else()
        message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
    endif()
elseif(USE_NVIDIA_BACKEND)    
    message(STATUS "Enabling NVIDIA backend")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}") # Default flags for NV backend
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda ") 
    if(USE_SM)
        message(STATUS "Building for SM_${USE_SM} architecture")
        string(APPEND CMAKE_CXX_FLAGS " -Xsycl-target-backend --cuda-gpu-arch=sm_${USE_SM}")
    endif()
elseif(USE_AMDHIP_BACKEND)
    message(STATUS "Enabling AMD HIP backend for ${USE_AMDHIP_BACKEND} AMD architecture")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}") # Default flags for AMD backend (gfx908 for MI100)
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${USE_AMDHIP_BACKEND}  ")
endif()

message(STATUS "Compilation flags set to: ${CMAKE_CXX_FLAGS}")

link_libraries(stdc++fs dnnl)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} sycl stdc++fs dnnl)
