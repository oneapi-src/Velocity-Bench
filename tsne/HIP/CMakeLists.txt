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

if(NOT DEFINED ROCM_PATH)
    if(NOT DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which HIP has been installed")
    else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
set(HIP_INCLUDE_DIRS  "${ROCM_PATH}/include"   ${HIP_INCLUDE_DIRS})
set(HIP_LIBRARIES     "${ROCM_PATH}/lib"       ${HIP_LIBRARIES})

set(DEF_WL_CXX_FLAGS           " -D__HIP_PLATFORM_AMD__ ")
set(DEF_GENERAL_CXX_FLAGS      " -O3 -std=c++17 -ffast-math ")
set(DEF_COMBINED_CXX_FLAGS     "${DEF_GENERAL_CXX_FLAGS} ${DEF_WL_CXX_FLAGS}")

# -DCMAKE_CXX_FLAGS=" -blah -blah " overrides the default flags (BOTH general and WL specific)
# -DOVERRIDE_GENERAL_CXX_FLAGS=" -blah -blah " overrides the general flags only (and not the workload specific flags)
# passing in both CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS is not allowed, in order to prevent ambiguity

if(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "" AND NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(FATAL_ERROR "Both  CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS cannot be passed in together")
elseif("${CMAKE_CXX_FLAGS}" STREQUAL "" AND "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags")
    set(CMAKE_CXX_FLAGS "${DEF_COMBINED_CXX_FLAGS}")
elseif(NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(STATUS "OVERRIDING GENERAL compilation flags")
    set(CMAKE_CXX_FLAGS "${OVERRIDE_GENERAL_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS ${DEF_WL_CXX_FLAGS})
elseif(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "OVERRIDING GENERAL and WORKLOAD SPECIFIC compilation flags")
endif()

find_package(HIP REQUIRED)

set(SOURCES
    # Utils
    ${CMAKE_SOURCE_DIR}/src/utils/debug_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/cuda_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/distance_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/math_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/matrix_broadcast_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/reduce_utils.cpp
    ${CMAKE_SOURCE_DIR}/../data/verify.cpp

    # Kernels
    ${CMAKE_SOURCE_DIR}/src/kernels/apply_forces.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/attr_forces.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/rep_forces.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/perplexity_search.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/nbodyfft.cpp

    # Method files
    ${CMAKE_SOURCE_DIR}/src/fit_tsne.cpp

    ${CMAKE_SOURCE_DIR}/src/exe/main.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/../data
    ${HIP_INCLUDE_DIRS}
)

add_executable(tsne ${SOURCES})

target_link_libraries(tsne -L${HIP_LIBRARIES} -lhipblas -lhipfft -lhipsparse)
