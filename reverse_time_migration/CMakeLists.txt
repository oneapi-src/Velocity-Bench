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

# Set minimum CMake version
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C++ standard and compiler options
find_program(BASH_EXECUTABLE NAMES bash REQUIRED)

# Intel Parallel Studio tools should be Sourced
# when using the cmake command or with the build script
if ("${USE_INTEL}" STREQUAL "")
    SET(USE_INTEL "NO")
endif ()

option(USE_DPC                  "Use DPC++ technology"                     OFF)
option(USE_CUDA                 "Use CUDA"                                 OFF)
option(USE_HIP                  "Use AMD"                                  OFF)

option(USE_INTEL_CPU            "Build AOT for Intel CPU"                   OFF)
option(GPU_AOT                  "Build AOT for Intel GPU"                   OFF)
option(USE_NVIDIA_BACKEND       "Use DPC++ technology with NVIDIA backend"  OFF)
option(USE_AMDHIP_BACKEND       "Use DPC++ technology with AMD HIP backend" OFF)
option(ENABLE_GPU_TIMINGS       "Show GPU timings at end of execution"      OFF)

set(USE_DEFAULT_FLAGS ON)
set(INTEL_GPU_CXX_FLAGS  "-O3 -fsycl " )
set(NVIDIA_GPU_CXX_FLAGS "-O3 -fsycl " ) # For DPC++ NVIDIA_BACKEND
set(AMD_GPU_CXX_FLAGS    "-O3 -fsycl " )
set(NVCC_GPU_CXX_FLAGS   "-O3 ")         # For NVCC 
set(ROCM_CXX_FLAGS       "-O3 ")         # For AMDHIP


message(STATUS "Use DPC++ ${USE_DPC} and Use CUDA ${USE_CUDA} and Use HIP ${USE_HIP}")

if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags for the application")
    string(APPEND CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}") # Default flags for NV backend
else()
    message(STATUS "OVERRIDING compilation flags")
    set(USE_DEFAULT_FLAGS OFF)
endif()

if(ENABLE_GPU_TIMINGS)
    message(STATUS "GPU Timings will be displayed")
    add_compile_options(-DENABLE_GPU_TIMINGS)
endif()

if (NOT USE_DPC AND NOT USE_CUDA AND NOT USE_HIP)
    message(FATAL_ERROR "Please specify a technology using the config.sh script")
elseif (USE_DPC AND USE_CUDA AND USE_HIP OR USE_DPC AND USE_CUDA OR USE_CUDA AND USE_HIP)
    message(FATAL_ERROR "Please specify only one technology using the config.sh script")
elseif (USE_DPC)
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
        string(APPEND CMAKE_CXX_FLAGS "  -ffast-math -mprefer-vector-width=512 -mfma -fsycl-targets=spir64_x86_64--linux \"-device avx512\" ")
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
        message(STATUS "Enabling AMD HIP backend for ${USE_HIP_BACKEND} AMD architecture")
        if(USE_DEFAULT_FLAGS)
            set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}")
        endif()
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${USE_AMDHIP_BACKEND}  ")
        add_compile_options(-DUSE_AMDHIP_BACKEND)
    endif()
elseif (USE_CUDA)
    set(CMAKE_CXX_FLAGS "${NVCC_GPU_CXX_FLAGS}")
    add_compile_options(-DENABLE_CUDA_LOGGING)
elseif (USE_HIP)
    set(CMAKE_CXX_FLAGS "${ROCM_CXX_FLAGS}")
    add_compile_options(-DENABLE_HIP_LOGGING)
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "" OR ${CMAKE_BUILD_TYPE} STREQUAL "NOMODE")
    message(STATUS "WORKING ON NO MODE")
elseif (${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
    message(STATUS "WORKING ON RELEASE MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
elseif (${CMAKE_BUILD_TYPE} STREQUAL "DEBUG" OR ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(STATUS "WORKING ON DEBUG MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
else ()
    message(FATAL_ERROR "Unrecognized build type")
endif ()


##############################################################################
# COMPILER DEFINITIONS
###############################################################################
if ("${DATA_PATH}" STREQUAL "")
    SET(DATA_PATH "./data")
endif ()
if ("${WRITE_PATH}" STREQUAL "")
    SET(WRITE_PATH "./results")
endif ()
if ("${WORKLOAD_PATH}" STREQUAL "")
    SET(WORKLOAD_PATH "./workloads/bp_model")
endif ()
if ("${COMPRESSION}" STREQUAL "")
    SET(COMPRESSION "NO")
endif ()

add_definitions(
        -DDATA_PATH="${DATA_PATH}"
        -DWRITE_PATH="${WRITE_PATH}"
        -DWORKLOAD_PATH="${WORKLOAD_PATH}")

##############################################################################
# PROJECT SETTINGS
##############################################################################

# Project name and language
project(Seismic-Toolbox LANGUAGES CXX VERSION 3.0.0)


##############################################################################
# LIBRARIES SETTINGS
##############################################################################

option(USE_MPI "Use MPI technology" OFF)
if (USE_MPI)
    add_definitions(-DUSING_MPI)

    message(STATUS "Using MPI shot distribution")
    find_package(MPI REQUIRED)

    include_directories(${MPI_INCLUDE_PATH})
endif ()

option(USE_OpenCV "Use OpenCV technology" OFF)
if (USE_OpenCV)
    message(STATUS "Using OpenCV technology")
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
else ()
    message(STATUS "No OpenCV used")
endif ()

##############################################################################
# HEADERS INCLUDE
##############################################################################

# Adding the headers directories needed for all projects.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/prerequisites)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/Helpers/MemoryManager/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/Helpers/Timer/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/seismic-io-framework)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/seismic-io-framework/Segy)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/seismic-io-framework/visualization)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/SeismicOperations/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/Thoth/include)
##############################################################################
# SUBDIRECTORIES ADDITION
##############################################################################

# TECHNOLOGY
if (USE_DPC)
    add_definitions(-DUSING_DPCPP)
elseif (USE_CUDA)
    add_definitions(-DUSING_CUDA -DOPERATIONS_LIB_BACKEND_ONEAPI_BACKEND_HPP) ### Adding -DOPS_LIB_BACKEND_API prevents CUDA including the oneAPI backend hpp
    find_package(CUDA REQUIRED) ### This needs to be here for some reason, not above
    set(KERNEL_SOURCES libs/SeismicOperations/src/components/concrete/cuda/model-handlers/SeismicModelHandler.cu
                       libs/SeismicOperations/src/components/concrete/cuda/boundary-managers/extensions/ZeroExtension.cu
                       libs/SeismicOperations/src/components/concrete/cuda/source-injectors/RickerSourceInjector.cu
                       libs/SeismicOperations/src/components/concrete/cuda/computation-kernels/isotropic/SecondOrderComputationKernel.cu
                       libs/SeismicOperations/src/components/concrete/cuda/trace-managers/SeismicTraceManager.cu
                       libs/SeismicOperations/src/components/concrete/cuda/migration-accommodators/CrossCorrelationKernel.cu
                       ) 
    message(STATUS "Found kernel sources : " ${KERNEL_SOURCES})
    if(USE_SM)
        message(STATUS "Building for SM_${USE_SM} compatibility")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${USE_SM},code=sm_${USE_SM} -std=c++17)
    endif()                          
    include_directories(${CUDA_TOOLKIT_INCLUDE})
elseif (USE_HIP)
    if (USE_ROCM_PATH)
        message(STATUS "Using custom ROCM PATH: " ${USE_ROCM_PATH})
        list(APPEND CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${USE_ROCM_PATH}/hip" "${USE_ROCM_PATH}")
    else()
        message(STATUS "Using default ROCM PATH /opt/rocm") 
        list(APPEND CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/rocm/hip" "/opt/rocm") 
    endif()

    find_package(HIP REQUIRED)
    add_definitions(-DUSING_AMD -DENABLE_HIP_LOGGING)
    message(STATUS "Compiling for AMD GPUs")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/Helpers)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/seismic-io-framework)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/SeismicOperations)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/Thoth)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

message(STATUS "Compilation flags set to: ${CMAKE_CXX_FLAGS}")
#message(STATUS "Flags : ${CMAKE_CXX_FLAGS}")
##############################################################################
# EXECUTABLES
##############################################################################

if (USE_CUDA)
    cuda_add_executable(Engine main_migration.cpp ${KERNEL_SOURCES})
    target_link_libraries(Engine SEISMIC-OPERATIONS SEISMIC-TOOLBOX ${CUDA_LIBRARIES})
    cuda_add_executable(Modeller main_modelling.cpp ${KERNEL_SOURCES})
    target_link_libraries(Modeller SEISMIC-OPERATIONS SEISMIC-TOOLBOX ${CUDA_LIBRARIES})
else ()
    add_executable(Engine main_migration.cpp)
    target_link_libraries(Engine SEISMIC-OPERATIONS SEISMIC-TOOLBOX)
    add_executable(Modeller main_modelling.cpp)
    target_link_libraries(Modeller SEISMIC-OPERATIONS SEISMIC-TOOLBOX)
endif()

##############################################################################
# EXAMPLES
##############################################################################

option(BUILD_EXAMPLES "Option to enable building example programs" OFF)
if (BUILD_EXAMPLES)
    message(STATUS "Building Seismic Toolbox Examples")
endif ()

##############################################################################
# TESTING
##############################################################################

option(BUILD_TESTS "Option to enable building tests" OFF)
if (BUILD_TESTS)
    message(STATUS "Building Seismic Toolbox Tests")
    add_subdirectory(tests)
    include(CTest)
    add_test(NAME SEISMIC-TOOLBOX-TESTS
            COMMAND seismic-toolbox-tests
            )
endif ()
