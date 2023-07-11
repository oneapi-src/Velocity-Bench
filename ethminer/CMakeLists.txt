### 
### Copyright (C) <2023> Intel Corporation
### 
### This program is free software; you can redistribute it and/or modify it
### under the terms of the GNU General Public License, as published
### by the Free Software Foundation; either version 2 of the License,
### or (at your option) any later version.
###  
### This program is distributed in the hope that it will be useful,
### but WITHOUT ANY WARRANTY; without even the implied warranty of
### MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
### GNU General Public License for more details.
###  
### You should have received a copy of the GNU General Public License
### along with this program; if not, see <http://www.gnu.org/licenses/>.
###  
### 
### SPDX-License-Identifier: GPL-2.0-or-later
### 
### 

cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED ENV{ethash_DIR})
    message(FATAL_ERROR "Before proceeding, please install the ethash v0.4.3 that is included with the source code. Then set ethash_DIR=/path/to/install/directory")
endif()

include(cmake/cable/bootstrap.cmake)

include(CableBuildInfo)

project(ethminer)
set(PROJECT_VERSION 0.19.0)

option(ETHASHCUDA "Build with CUDA mining"             OFF)
option(ETHASHHIP  "Build with HIP mining"              OFF)
option(ETHASHSYCL "Build with sycl mining"              ON)
option(ETHDBUS    "Build with D-Bus support"           OFF)
option(APICORE    "Build with API Server support"       ON)
option(DEVBUILD   "Log developer metrics"              OFF)

option(GPU_AOT                 "Build AOT for Intel GPU"             OFF)
option(USE_NVIDIA_BACKEND      "Build for NVIDIA backend"            OFF)
option(USE_AMDHIP_BACKEND      "Build for AMD HIP backend"           OFF)
option(USE_LOOP_UNROLLING      "Build with loop unrolling SYCL only" OFF)
option(USE_SM                  "Build for CUDA architecture"     )

set(USE_DEFAULT_FLAGS ON)
set(INTEL_GPU_CXX_FLAGS  " -O2 -fsycl ")
set(NVIDIA_GPU_CXX_FLAGS " -O3 -fsycl ")
set(AMD_GPU_CXX_FLAGS    " -O3 -fsycl ")
set(ROCM_CXX_FLAGS       " -O3" )
set(NVCC_CXX_FLAGS       " -O3 ")

include_directories(${OPENSSL_INCLUDE_DIR} ${jsoncpp_DIR}/../../../include)

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
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_gen -Xs \"-device 0x0bd5 -revision_id 0x2f\" -Xs \"-options -ze-opt-large-register-file\" ") # Default flag for pvc 
    else()
        message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
    endif()
elseif(USE_NVIDIA_BACKEND)    
    message(STATUS "Enabling NVIDIA backend")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}") # Default flags for NV backend
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda ") # -O3 will be used, even though -O2 was set earlier 
    if(USE_SM)
        message(STATUS "Building for SM_${USE_SM} architecture")
        string(APPEND CMAKE_CXX_FLAGS " -Xsycl-target-backend --cuda-gpu-arch=sm_${USE_SM}")
    endif()
elseif(USE_AMDHIP_BACKEND)
    message(STATUS "Enabling AMD HIP backend for ${USE_AMDHIP_BACKEND} AMD architecture")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}")
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${USE_AMDHIP_BACKEND}  ")
    add_compile_options(-DUSE_AMDHIP_BACKEND)
endif()

if(USE_AMDHIP_BACKEND)
    message(STATUS "Using 64 thread shuffle function code path")
else()
    message(STATUS "Using 32 thread shuffle function code path")
endif()

if (ETHASHCUDA)
    if (USE_DEFAULT_FLAGS)
        message(STATUS "Using NVIDIA CUDA flags")
        set(CMAKE_CXX_FLAGS "${NVCC_CXX_FLAGS}")
    endif()
endif()

if (ETHASHHIP)
    if (USE_DEFAULT_FLAGS)
        message(STATUS "Using ROCM flags")
        set(CMAKE_CXX_FLAGS "${ROCM_CXX_FLAGS}")
    endif()
    
    if (USE_ROCM_PATH) 
        message(STATUS "Using custom ROCM PATH: " ${USE_ROCM_PATH})
        list(APPEND CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${USE_ROCM_PATH}/hip" "${USE_ROCM_PATH}")
    else()
        message(STATUS "Using default ROCM PATH /opt/rocm") 
        list(APPEND CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/rocm/hip" "/opt/rocm")
    endif()
    find_package(hip REQUIRED)
endif()

# propagates CMake configuration options to the compiler
function(configureProject)
    if (ETHASHCUDA)
        add_definitions(-DETH_ETHASHCUDA)
    endif()
    if (ETHASHHIP)
        add_definitions(-DETH_ETHASHHIP)
    endif()
    if (ETHASHSYCL)
        add_definitions(-DETH_ETHASHSYCL)
        if (USE_LOOP_UNROLLING)
            add_definitions(-DUSE_LOOP_UNROLLING)
        endif()
    endif()
    if (ETHDBUS)
        add_definitions(-DETH_DBUS)
    endif()
    if (APICORE)
        add_definitions(-DAPI_CORE)
    endif()
    if (DEVBUILD)
        add_definitions(-DDEV_BUILD)
    endif()
endfunction()

find_package(Boost CONFIG REQUIRED COMPONENTS system filesystem thread)

find_package(jsoncpp CONFIG REQUIRED)

find_package(ethash REQUIRED) 

configureProject()

message("----------------------------------------------------------------------------")
message("-- CMake ${CMAKE_VERSION}")
message("-- Build ${CMAKE_BUILD_TYPE} / ${CMAKE_SYSTEM_NAME}")
message("----------------------------------------------------------------- components")
message("-- ETHASHCUDA       Build CUDA components                        ${ETHASHCUDA}")
message("-- ETHASHHIP        Build HIP components                         ${ETHASHHIP}")
message("-- ETHASHSYCL       Build SYCL components                        ${ETHASHSYCL}")
message("-- ETHDBUS          Build D-Bus components                       ${ETHDBUS}")
message("-- APICORE          Build API Server components                  ${APICORE}")
message("-- DEVBUILD         Build with dev logging                       ${DEVBUILD}")
message("----------------------------------------------------------------------------")
message("")

include(EthCompilerSettings)
if(UNIX AND NOT APPLE)
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")
endif()

cable_add_buildinfo_library(PROJECT_NAME ${PROJECT_NAME})

add_subdirectory(libdevcore)
add_subdirectory(libethcore)
add_subdirectory(libhwmon)
add_subdirectory(libpoolprotocols)

message(STATUS "BOOST  : ${Boost_INCLUDE_DIRS}")
message(STATUS "OPENSSL: ${OPENSSL_INCLUDE_DIR}")
message(STATUS "JSON   : ${jsoncpp_DIR}/../../../include")
include_directories(${Boost_INCLUDE_DIRS}/include ${OPENSSL_INCLUDE_DIR} ${jsoncpp_DIR}/../../../include)

if (ETHASHCUDA)
    add_subdirectory(libethash-cuda)
    if (USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${NVCC_CXX_FLAGS}")
    endif()
endif ()
if (ETHASHHIP)
    add_subdirectory(libethash-hip)
endif ()
if (ETHASHSYCL)
    add_subdirectory(libethash-sycl)
endif ()
if (APICORE)
    add_subdirectory(libapicore)
endif()

add_subdirectory(ethminer)

# Output the compiler flags that were constructed for visual inspection
message(STATUS "Compilation flags set to: ${CMAKE_CXX_FLAGS}")


if(WIN32)
    set(CPACK_GENERATOR ZIP)
else()
    set(CPACK_GENERATOR TGZ)
endif()
set(CPACK_PACKAGE_FILE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY FALSE)
include(CPack)
