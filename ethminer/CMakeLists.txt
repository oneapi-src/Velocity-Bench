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
option(USE_AMD_BACKEND         "Build for AMD HIP backend"           OFF)
option(USE_LOOP_UNROLLING      "Build with loop unrolling SYCL only" OFF)
option(USE_SM                  "Build for CUDA architecture"     )

### SYCL RELATED FLAGS START HERE
set(DEF_INTEL_GENERAL_CXX_FLAGS  " -O2 -fsycl ")
set(DEF_NVIDIA_GENERAL_CXX_FLAGS " -O3 -fsycl ")
set(DEF_AMD_GENERAL_CXX_FLAGS    " -O3 -fsycl ")

set(DEF_INTEL_WL_CXX_FLAGS  " ")
set(DEF_NVIDIA_WL_CXX_FLAGS " ")
set(DEF_AMD_WL_CXX_FLAGS    " ")

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
### END OF SYCL RELATED FLAGS

### NVCC RELATED FLAGS START HERE
set(NVCC_WL_CXX_FLAGS       " ")
set(NVCC_DEF_CXX_FLAGS      " -O3 ")
set(NVCC_DEF_COMBINED_FLAGS "${NVCC_WL_CXX_FLAGS} ${NVCC_DEF_CXX_FLAGS} ")
### END OF NVCC RELATED FLAGS


### ROCM RELATED FLAGS START HERE 
set(ROCM_WL_CXX_FLAGS        "  ")
set(ROCM_DEF_CXX_FLAGS       " -O3 -D__HIP_PLATFORM_AMD__  " )
set(ROCM_DEF_COMBINED_FLAGS  " ${ROCM_WL_CXX_FLAGS} ${ROCM_DEF_CXX_FLAGS} ")
### END OF ROCM RELATED FLAGS

include_directories(${OPENSSL_INCLUDE_DIR} ${jsoncpp_DIR}/../../../include)

# JIT compilation 
if(GPU_AOT)
    set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}")
    if( (${GPU_AOT} STREQUAL "pvc") OR (${GPU_AOT} STREQUAL "PVC") )
        message(STATUS "Enabling Intel GPU AOT compilation for ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_gen -Xs \"-device 0x0bd5 -revision_id 0x2f\" -Xs \"-options -ze-opt-large-register-file\" ") # Default flag for pvc 
    else()
        message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
    endif()
elseif(USE_NVIDIA_BACKEND)    
    message(STATUS "Enabling NVIDIA backend")
    set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda ") # -O3 will be used, even though -O2 was set earlier 
    if(USE_SM)
        message(STATUS "Building for SM_${USE_SM} architecture")
        string(APPEND CMAKE_CXX_FLAGS " -Xsycl-target-backend --cuda-gpu-arch=sm_${USE_SM}")
    endif()
elseif(USE_AMD_BACKEND)
    if ("${USE_AMD_ARCH}" STREQUAL "")
        message(FATAL_ERROR "Must specify AMD arch, e.g., -DUSE_AMD_ARCH=gfx90a")
    endif()
    message(STATUS "Enabling AMD backend for ${USE_AMD_ARCH} AMD architecture")
    set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${USE_AMD_ARCH}  ")
    add_compile_options(-DUSE_AMD_BACKEND)
endif()

if(USE_AMD_BACKEND)
    message(STATUS "Using 64 thread shuffle function code path")
else()
    message(STATUS "Using 32 thread shuffle function code path")
endif()

if (ETHASHCUDA)
    message(STATUS "Compiling for NVCC")
    if(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "" AND NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
        message(FATAL_ERROR "Both  CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS cannot be passed in together")
    elseif("${CMAKE_CXX_FLAGS}" STREQUAL "" AND "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
        message(STATUS "Using DEFAULT compilation flags")
        set(CMAKE_CXX_FLAGS "${NVCC_DEF_COMBINED_FLAGS}")
    elseif(NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
        message(STATUS "OVERRIDING GENERAL compilation flags")
        set(CMAKE_CXX_FLAGS "${OVERRIDE_GENERAL_CXX_FLAGS}")
        string(APPEND CMAKE_CXX_FLAGS ${NVCC_WL_CXX_FLAGS})
    elseif(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "")
        message(STATUS "OVERRIDING GENERAL and WORKLOAD SPECIFIC compilation flags")
    endif()
endif()

if (ETHASHHIP)
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
