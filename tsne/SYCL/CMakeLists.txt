 # Modifications Copyright (C) 2023 Intel Corporation
 #
 # Redistribution and use in source and binary forms, with or without modification,
 # are permitted provided that the following conditions are met:
 #
 # 1. Redistributions of source code must retain the above copyright notice,
 #    this list of conditions and the following disclaimer.
 # 2. Redistributions in binary form must reproduce the above copyright notice,
 #   this list of conditions and the following disclaimer in the documentation
 #   and/or other materials provided with the distribution.
 # 3. Neither the name of the copyright holder nor the names of its contributors
 #   may be used to endorse or promote products derived from this software
 #   without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 # THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 # BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 # OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 # OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 # OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 # WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 # OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 # EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
 #
 # SPDX-License-Identifier: BSD-3-Clause
 #

cmake_minimum_required(VERSION 3.10)

project(tsne LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_KERNEL_PROFILING "Build using kernel profiling" OFF)
option(GPU_AOT                 "Build AOT for Intel GPU"      OFF)
option(USE_NVIDIA_BACKEND      "Build for NVIDIA backend"     OFF)
option(USE_AMDHIP_BACKEND      "Build for AMD HIP backend"    OFF)
option(USE_SM                  "Build for specific SM"        OFF)

if(ENABLE_KERNEL_PROFILING)
    message("-- Enabling kernel profiling")
    add_compile_options(-DENABLE_KERNEL_PROFILING)
endif()

set(DEF_INTEL_WL_CXX_FLAGS  " ")
set(DEF_NVIDIA_WL_CXX_FLAGS " ")
set(DEF_AMD_WL_CXX_FLAGS    " ")

set(DEF_INTEL_GENERAL_CXX_FLAGS  " -O2 -std=c++17 -fsycl -ffast-math -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-local-typedef ")
set(DEF_NVIDIA_GENERAL_CXX_FLAGS " -O3 -std=c++17 -fsycl -ffast-math -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-local-typedef ")
set(DEF_AMD_GENERAL_CXX_FLAGS    " -O3 -std=c++17 -fsycl -ffast-math -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-local-typedef ")

# -DCMAKE_CXX_FLAGS=" -blah -blah " overrides the default flags (BOTH general and WL specific)
# -DOVERRIDE_GENERAL_CXX_FLAGS=" -blah -blah " overrides the general flags only (and not the workload specific flags)
# passing in both CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS is not allowed, in order to prevent ambiguity

if(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "" AND NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(FATAL_ERROR "Both  CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS cannot be passed in together")
elseif("${CMAKE_CXX_FLAGS}" STREQUAL "" AND "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags")
    set(INTEL_GPU_CXX_FLAGS        "${DEF_INTEL_GENERAL_CXX_FLAGS}   ${DEF_INTEL_WL_CXX_FLAGS}")
    set(NVIDIA_GPU_CXX_FLAGS       "${DEF_NVIDIA_GENERAL_CXX_FLAGS}  ${DEF_NVIDIA_WL_CXX_FLAGS}")
    set(AMD_GPU_CXX_FLAGS          "${DEF_AMD_GENERAL_CXX_FLAGS}     ${DEF_AMD_WL_CXX_FLAGS}")
elseif(NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(STATUS "OVERRIDING GENERAL compilation flags")
    set(INTEL_GPU_CXX_FLAGS        "${OVERRIDE_GENERAL_CXX_FLAGS}    ${DEF_INTEL_WL_CXX_FLAGS}")
    set(NVIDIA_GPU_CXX_FLAGS       "${OVERRIDE_GENERAL_CXX_FLAGS}    ${DEF_NVIDIA_WL_CXX_FLAGS}")
    set(AMD_GPU_CXX_FLAGS          "${OVERRIDE_GENERAL_CXX_FLAGS}    ${DEF_AMD_WL_CXX_FLAGS}")
elseif(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "OVERRIDING GENERAL and WORKLOAD SPECIFIC compilation flags")
    set(INTEL_GPU_CXX_FLAGS        "${CMAKE_CXX_FLAGS}")
    set(NVIDIA_GPU_CXX_FLAGS       "${CMAKE_CXX_FLAGS}")
    set(AMD_GPU_CXX_FLAGS          "${CMAKE_CXX_FLAGS}")
endif()

if(GPU_AOT)
    message(STATUS "Enabling INTEL backend")
    set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}")
    if((${GPU_AOT} STREQUAL "pvc") OR (${GPU_AOT} STREQUAL "PVC"))
        message(STATUS "Enabling Intel GPU AOT compilation for ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_gen -Xs \"-device 0x0bd5 -revision_id 0x2f\" ")
    else()
        message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
    endif()
elseif(USE_NVIDIA_BACKEND)
    message(STATUS "Enabling NVIDIA backend")
    set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}")
    if(USE_SM)
        message("-- Building for SM_${USE_SM} compatibility")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_${USE_SM} ")
    else()
        message("-- Building for SM_80 compatibility (DEFAULT)")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 ")
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_NVIDIA_BACKEND")
elseif(USE_AMDHIP_BACKEND)
    message(STATUS "Enabling AMD HIP backend for ${USE_AMDHIP_BACKEND} AMD architecture")
    set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${USE_AMDHIP_BACKEND} ")
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_AMDHIP_BACKEND")
else()
    # JIT case
    message(STATUS "Enabling INTEL backend")
    set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}")
endif()

message(STATUS "CXX Compilation flags set to: ${CMAKE_CXX_FLAGS}")

if(NOT USE_NVIDIA_BACKEND)
    set(MKL_LINK static)
    set(MKL_THREADING sequential)
    find_package(MKL CONFIG REQUIRED HINTS "$ENV{MKLROOT}/lib/cmake/mkl")
else()
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()
# Project Setup
#-------------------------------------------------------------------------------
set(SOURCES
    # # Utils
    ${CMAKE_SOURCE_DIR}/src/utils/debug_utils.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/cuda_utils.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/distance_utils.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/math_utils.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/matrix_broadcast_utils.dp.cpp
    # ${CMAKE_SOURCE_DIR}/src/utils/reduce_utils.dp.cpp
    ${CMAKE_SOURCE_DIR}/../data/verify.cpp

    # # Kernels
    ${CMAKE_SOURCE_DIR}/src/kernels/apply_forces.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/attr_forces.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/rep_forces.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/perplexity_search.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/nbodyfft.dp.cpp

    # Method files
    ${CMAKE_SOURCE_DIR}/src/fit_tsne.dp.cpp

    ${CMAKE_SOURCE_DIR}/src/exe/main.dp.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}/src/
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/../data
)

add_executable(tsne ${SOURCES})

if(NOT USE_NVIDIA_BACKEND)
    target_compile_options(tsne PUBLIC $<TARGET_PROPERTY:MKL::MKL_DPCPP,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(tsne PUBLIC $<TARGET_PROPERTY:MKL::MKL_DPCPP,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(tsne PUBLIC $<LINK_ONLY:MKL::MKL_DPCPP>)
else()
    target_link_libraries(tsne ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_cusparse_LIBRARY})
endif()

