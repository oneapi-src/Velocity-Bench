 # Modifications Copyright (C) 2023 Intel Corporation
 #
 # This Program is subject to the terms of the GNU General Public License v3.0 only.​
 # If a copy of the license was not distributed with this file, ​
 # you can obtain one at https://spdx.org/licenses/GPL-3.0-only.html
 #​
 #
 # SPDX-License-Identifier: GPL-3.0-only
 #

cmake_minimum_required(VERSION 3.10)

project(SeisAcoMod2D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MPI_HOME)
    message(STATUS "Using provided MPI_HOME=${MPI_HOME}")
else()
    # set(MPI_HOME "/opt/intel/oneapi/mpi/latest")
    set(MPI_HOME "/usr/lib/x86_64-linux-gnu/openmpi")
    message(STATUS "Using default MPI_HOME=${MPI_HOME}")
endif()

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
set(DEF_GENERAL_CXX_FLAGS      " -O3 -std=c++17 ")
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

set(SOURCES
    ${CMAKE_SOURCE_DIR}/../common/main.cpp
    ${CMAKE_SOURCE_DIR}/../common/json_parser.cpp
    ${CMAKE_SOURCE_DIR}/../common/modelling_worker.cpp
    ${CMAKE_SOURCE_DIR}/../common/modelling_master.cpp
    ${CMAKE_SOURCE_DIR}/../common/read_receiver.cpp
    ${CMAKE_SOURCE_DIR}/../common/create_geometry.cpp
    ${CMAKE_SOURCE_DIR}/../common/workload.cpp
    ${CMAKE_SOURCE_DIR}/../common/model.cpp
    ${CMAKE_SOURCE_DIR}/../common/check_stability.cpp

    ${CMAKE_SOURCE_DIR}/src/gpu_modelling_kernels.hp.cpp
    ${CMAKE_SOURCE_DIR}/src/gpumodelling.hp.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}/../common/
    ${CMAKE_SOURCE_DIR}/src/
    ${MPI_HOME}/include/
    ${HIP_INCLUDE_DIRS}
)

add_executable(SeisAcoMod2D ${SOURCES})

target_link_libraries(SeisAcoMod2D -L${HIP_LIBRARIES} -L${MPI_HOME}/lib -lmpi_cxx -lmpi)
