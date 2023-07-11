# ModificationsCopyright(C) 2023 Intel Corporation

# Redistributionand use in source and binary forms, with or without modification,

# arepermitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,

# thislist of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,

# thislist of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors

# maybe used to endorse or promote products derived from this software

# withoutspecific prior written permission.

# THISSOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"

# ANDANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,

# THEIMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE

# AREDISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS

# BELIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,

# ORCONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT

# OFSUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;

# ORBUSINESS INTERRUPTION)
# HOWEVERCAUSED AND ON ANY THEORY OF LIABILITY,

# WHETHERIN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE

# OROTHERWISE)
# ARISINGIN ANY WAY OUT OF THE USE OF THIS SOFTWARE,

# EVEN

# IFADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 2.8)
project(qs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_VERBOSE_MAKEFILE on)
option(ENABLE_KERNEL_PROFILING "Build using kernel profiling" OFF)
option(GPU_AOT "Build AOT for Intel GPU" OFF)
option(USE_NVIDIA_BACKEND "Build for NVIDIA backend" OFF)
option(USE_AMDHIP_BACKEND "Build for AMD HIP backend" OFF)
option(USE_SM "Specifies which streaming multiprocessor architecture to use" OFF)

# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -std=c++11 -O0 -Wall -g -lpthread -ggdb -pg")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -fsycl -Wall -g -lpthread -ggdb -pg -std=c++11")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -fsycl -Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG ${CXXFLAGS} "-fsycl -Wall -O1 -Wextra -pg")
set(CMAKE_CXX_FLAGS_RELEASE ${CXXFLAGS} "-fsycl -Wall -O3 -Wextra -ffp-contract=fast")

include_directories(
    ${CMAKE_SOURCE_DIR}/src/
)

set(SRC_LIST
    ${CMAKE_SOURCE_DIR}/src/CoralBenchmark.cc
    ${CMAKE_SOURCE_DIR}/src/CycleTracking.cc
    ${CMAKE_SOURCE_DIR}/src/DecompositionObject.cc
    ${CMAKE_SOURCE_DIR}/src/DirectionCosine.cc.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/EnergySpectrum.cc
    ${CMAKE_SOURCE_DIR}/src/GlobalFccGrid.cc.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/GridAssignmentObject.cc.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/InputBlock.cc
    ${CMAKE_SOURCE_DIR}/src/MC_Base_Particle.cc
    ${CMAKE_SOURCE_DIR}/src/MC_Domain.cc.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/MC_Fast_Timer.cc.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/MC_Particle_Buffer.cc
    ${CMAKE_SOURCE_DIR}/src/MeshPartition.cc
    ${CMAKE_SOURCE_DIR}/src/MonteCarlo.cc.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/MpiCommObject.cc
    ${CMAKE_SOURCE_DIR}/src/Parameters.cc
    ${CMAKE_SOURCE_DIR}/src/ParticleVault.cc
    ${CMAKE_SOURCE_DIR}/src/ParticleVaultContainer.cc
    ${CMAKE_SOURCE_DIR}/src/PopulationControl.cc.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/SharedMemoryCommObject.cc
    ${CMAKE_SOURCE_DIR}/src/Tallies.cc
    ${CMAKE_SOURCE_DIR}/src/cmdLineParser.cc
    ${CMAKE_SOURCE_DIR}/src/cudaFunctions.cc.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/initMC.cc.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/main.cc.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/parseUtils.cc
    ${CMAKE_SOURCE_DIR}/src/utils.cc
    ${CMAKE_SOURCE_DIR}/src/utilsMpi.cc.dp.cpp
    ${CMAKE_SOURCE_DIR}/src/Random.cc
)

# Declare the executable target built from your sources
# add_executable(gpu_example test.cc)

# enable kernel profiling on demand
if(GEN9)
    message("-- NOTE: Building for GEN9 archetecture")
    add_definitions(-DGEN9)
endif()

if(ENABLE_KERNEL_PROFILING)
    message("-- NOTE: Enabling Kernel time measurement")
    add_definitions(-DENABLE_KERNEL_PROFILING)
endif()

if(DEBUG)
    message("-- NOTE: Enabling debug mode")
    add_definitions(-DDEBUG)
endif()

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Release")
    message("-- NOTE: setting build_type to release")
    SET(CMAKE_BUILD_TYPE "Release")
else()
    message("-- NOTE: setting build_type to debug")
    SET(CMAKE_BUILD_TYPE "Debug")
endif()

set(USE_DEFAULT_FLAGS ON)
set(INTEL_GPU_CXX_FLAGS " -O2 -std=c++17 -DRUN_ON_GPU=1 -DHAVE_SYCL=1")
set(NVIDIA_GPU_CXX_FLAGS " -O3 -fsycl -DHAVE_SYCL")
set(AMD_GPU_CXX_FLAGS " -O0 -fsycl -DUNIFIED_HOST=1 -DHAVE_SYCL")

if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags for the application")
    string(APPEND CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}") # Default flags for NV backend
else()
    message(STATUS "OVERRIDING compilation flags")
    set(USE_DEFAULT_FLAGS OFF)
endif()

if(GPU_AOT)
    if((${GPU_AOT} STREQUAL "pvc") OR(${GPU_AOT} STREQUAL "PVC"))
        message(STATUS "Enabling Intel GPU AOT compilation for ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_gen -Xs \"-device 0x0bd5 -revision_id 0x2f\" -Xs \"-options -ze-opt-large-register-file\" ")
    else()
        message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
    endif()
elseif(USE_NVIDIA_BACKEND)
    message(STATUS "Enabling NVIDIA backend")

    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}") # Default flags for NV backend
    endif()

    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_${USE_SM} ") # -O3 will be used, even though -O2 was set earlier
elseif(USE_AMDHIP_BACKEND)
    message(STATUS "Enabling AMD HIP backend for ${USE_AMDHIP_BACKEND} AMD architecture")

    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}")
    endif()

    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${USE_AMDHIP_BACKEND}  ")
endif()

add_executable(qs ${SRC_LIST})
target_link_libraries(qs sycl stdc++ stdc++fs)
