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


set(DEF_INTEL_WL_CXX_FLAGS  " ")
set(DEF_NVIDIA_WL_CXX_FLAGS " ")
set(DEF_AMD_WL_CXX_FLAGS    " -D__HIP_PLATFORM_AMD__ ")

set(DEF_INTEL_GENERAL_CXX_FLAGS  " -O2 -fsycl -ffast-math ")
set(DEF_NVIDIA_GENERAL_CXX_FLAGS " -O2 -fsycl -ffast-math ")
set(DEF_AMD_GENERAL_CXX_FLAGS    " -O2 -fsycl -ffast-math ")


# -DCMAKE_CXX_FLAGS=" -blah -blah " overrides the default flags (BOTH general and WL specific)
# -DOVERRIDE_GENERAL_CXX_FLAGS=" -blah -blah " overrides the general flags only (and not the workload specific flags)
# passing in both CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS is not allowed, in order to prevent ambiguity

#set(USE_DEFAULT_FLAGS OFF)
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


add_compile_options(-DRUN_ON_GPU)
#add_compile_options(-DKERNEL_USE_PROFILE "0")
if(DEVICE_TIMER)
    message(STATUS "Enabling Device Timer")
    add_compile_options(-DDEVICE_TIMER)
endif()

# JIT compilation 
if(GPU_AOT) 
    message(STATUS "Enabling INTEL backend")
    set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}")
    if( (${GPU_AOT} STREQUAL "pvc") OR (${GPU_AOT} STREQUAL "PVC") )
        message(STATUS "Enabling Intel GPU AOT compilation for ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_gen -Xs \"-device 0x0bd5 -revision_id 0x2f\" ") 
    else()
        message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
    endif()
elseif(USE_NVIDIA_BACKEND)    
    message(STATUS "Enabling NVIDIA backend")
    set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda ") 
    if(USE_SM)
        message(STATUS "Building for SM_${USE_SM} architecture")
        string(APPEND CMAKE_CXX_FLAGS " -Xsycl-target-backend --cuda-gpu-arch=sm_${USE_SM}")
    endif()
elseif(USE_AMDHIP_BACKEND)
    message(STATUS "Enabling AMD HIP backend for ${USE_AMDHIP_BACKEND} AMD architecture")
    set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${USE_AMDHIP_BACKEND}  ")
else()
    # JIT case
    message(STATUS "Enabling INTEL backend")
    set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}")
endif()

message(STATUS "Compilation flags set to: ${CMAKE_CXX_FLAGS}")

link_libraries(stdc++fs dnnl)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} sycl stdc++fs dnnl)
