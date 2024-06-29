# Modifications Copyright (C) 2023 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
# OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

# SPDX-License-Identifier: MIT

CMAKE_MINIMUM_REQUIRED(VERSION 3.20 FATAL_ERROR)
PROJECT(voxelizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17) # SYCL code requires this
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Enable modern C++ standards

option(ENABLE_KERNEL_PROFILING "Build using kernel profiling" OFF)
option(GPU_AOT "Build AOT for Intel GPU" OFF)
option(USE_NVIDIA_BACKEND "Build for NVIDIA backend" OFF)
option(USE_AMDHIP_BACKEND "Build for AMD HIP backend" OFF)
option(USE_INTEL_CPU "Build for INTEL CPU" OFF)
option(USE_SM "Specifies which streaming multiprocessor architecture to use")

option(Trimesh2_LINK_DIR "Path to Trimesh2 library dir")
option(Trimesh2_INCLUDE_DIR "Path to Trimesh2 includes")

include_directories(
    ${CMAKE_SOURCE_DIR}/src/
)

find_package(glm REQUIRED)
message(STATUS "glm library status:")
message(STATUS "    version: ${glm_VERSION}")
message(STATUS "    libraries: ${GLM_LIBRARIES}")
message(STATUS "    include path: ${GLM_INCLUDE_DIRS}")

# FIND_PACKAGE(OpenMP REQUIRED)
# if (OPENMP_FOUND)
# message("openmp found")
# else()
# message("openmp not found")
# endif()

# message(STATUS "OpenMP library status:")
# message(STATUS "    version: ${OpenMP_VERSION}")
# message(STATUS "    libraries: ${OpenMP_LIBS}")
# message(STATUS "    include path: ${OpenMP_INCLUDE_DIRS}")
SET(VOXELIZER_EXECUTABLE voxelizer_sycl)

IF(NOT Trimesh2_INCLUDE_DIR)
    MESSAGE(FATAL_ERROR "You need to set variable Trimesh2_INCLUDE_DIR")
ENDIF()

FIND_FILE(Trimesh2_TriMesh_h TriMesh.h ${Trimesh2_INCLUDE_DIR})

IF(NOT Trimesh2_TriMesh_h)
    message(FATAL_ERROR "Can't find TriMesh.h in ${Trimesh2_INCLUDE_DIR}")
ENDIF()

MARK_AS_ADVANCED(Trimesh2_TriMesh_h)

IF(NOT Trimesh2_LINK_DIR)
    MESSAGE(FATAL_ERROR "You need to set variable Trimesh2_LINK_DIR")
ENDIF()

IF(NOT EXISTS "${Trimesh2_LINK_DIR}")
    MESSAGE(FATAL_ERROR "Trimesh2 library dir does not exist")
ENDIF()

FIND_LIBRARY(Trimesh2_LIBRARY trimesh ${Trimesh2_LINK_DIR})

IF(NOT Trimesh2_LIBRARY)
    message(SEND_ERROR "Can't find libtrimesh.a in ${Trimesh2_LINK_DIR}")
ENDIF()

MARK_AS_ADVANCED(Trimesh2_LIBRARY)

MESSAGE(STATUS "Found Trimesh2 include: ${Trimesh2_TriMesh_h}")
MESSAGE(STATUS "Found Trimesh2 lib: ${Trimesh2_LIBRARY}")

SET(SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/util_cuda.cpp.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/util_io.cpp.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_voxelizer.cpp.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/voxelize.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/thrust_operations.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/voxelize_solid.dp.cpp
)

if(USE_NVIDIA_BACKEND)
    message(STATUS "Nvidia backend")
    add_compile_options(-DUSE_NVIDIA_BACKEND)
endif()

if(USE_AMDHIP_BACKEND)
    message(STATUS "AMD backend")
    add_compile_options(-DUSE_AMDHIP_BACKEND)
endif()

set(DEF_INTEL_WL_CXX_FLAGS " -fopenmp ")
set(DEF_NVIDIA_WL_CXX_FLAGS " ")
set(DEF_AMD_WL_CXX_FLAGS " -D__HIP_PLATFORM_AMD__ ")

set(DEF_INTEL_GENERAL_CXX_FLAGS " -O3 -fsycl -ffast-math ")
set(DEF_NVIDIA_GENERAL_CXX_FLAGS " -O3 -fsycl -ffast-math ")
set(DEF_AMD_GENERAL_CXX_FLAGS " -O3 -fsycl -ffast-math ")

# -DCMAKE_CXX_FLAGS=" -blah -blah " overrides the default flags (BOTH general and WL specific)
# -DOVERRIDE_GENERAL_CXX_FLAGS=" -blah -blah " overrides the general flags only (and not the workload specific flags)
# passing in both CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS is not allowed, in order to prevent ambiguity
if(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "" AND NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(FATAL_ERROR "Both  CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS cannot be passed in together")
elseif("${CMAKE_CXX_FLAGS}" STREQUAL "" AND "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags")
    set(INTEL_GPU_CXX_FLAGS "${DEF_INTEL_GENERAL_CXX_FLAGS}   ${DEF_INTEL_WL_CXX_FLAGS}")
    set(NVIDIA_GPU_CXX_FLAGS "${DEF_NVIDIA_GENERAL_CXX_FLAGS}  ${DEF_NVIDIA_WL_CXX_FLAGS}")
    set(AMD_GPU_CXX_FLAGS "${DEF_AMD_GENERAL_CXX_FLAGS}     ${DEF_AMD_WL_CXX_FLAGS}")
elseif(NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(STATUS "OVERRIDING GENERAL compilation flags")
    set(INTEL_GPU_CXX_FLAGS "${OVERRIDE_GENERAL_CXX_FLAGS}    ${DEF_INTEL_WL_CXX_FLAGS}")
    set(NVIDIA_GPU_CXX_FLAGS "${OVERRIDE_GENERAL_CXX_FLAGS}    ${DEF_NVIDIA_WL_CXX_FLAGS}")
    set(AMD_GPU_CXX_FLAGS "${OVERRIDE_GENERAL_CXX_FLAGS}    ${DEF_AMD_WL_CXX_FLAGS}")
elseif(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "OVERRIDING GENERAL and WORKLOAD SPECIFIC compilation flags")
    set(INTEL_GPU_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(NVIDIA_GPU_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(AMD_GPU_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# JIT compilation
if(GPU_AOT)
    message(STATUS "Enabling INTEL backend")
    set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}")

    if((${GPU_AOT} STREQUAL "pvc") OR(${GPU_AOT} STREQUAL "PVC"))
        message(STATUS "Enabling Intel GPU AOT compilation for ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_gen -Xs \"-device 0x0bd5 -revision_id 0x2f\" -Xs \"-options -ze-opt-large-register-file\" ")
    else()
        message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
    endif()
elseif(USE_NVIDIA_BACKEND)
    message(STATUS "Enabling NVIDIA backend")
    set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_${USE_SM}")
elseif(USE_AMDHIP_BACKEND)
    message(STATUS "Enabling AMD HIP backend for ${USE_AMDHIP_BACKEND} AMD architecture")
    set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${USE_AMDHIP_BACKEND}  ")
else()
    message(STATUS "Enabling INTEL backend")
    set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}")
endif()

# Output the compiler flags that were constructed for visual inspection
message(STATUS "Compilation flags set to: ${CMAKE_CXX_FLAGS}")

ADD_EXECUTABLE(${VOXELIZER_EXECUTABLE} ${SOURCES})

TARGET_COMPILE_FEATURES(${VOXELIZER_EXECUTABLE} PRIVATE cxx_std_17)
TARGET_INCLUDE_DIRECTORIES(${VOXELIZER_EXECUTABLE} PRIVATE ${Trimesh2_INCLUDE_DIR} ${GLM_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(${VOXELIZER_EXECUTABLE} PRIVATE ${Trimesh2_LIBRARY} PRIVATE glm::glm stdc++ stdc++fs)
