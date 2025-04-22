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
project(dl-cifar_sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)           # SYCL code requires this
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Enable modern C++ standards
set(CMAKE_CXX_EXTENSIONS OFF)        # Use -std, not -gnu


option(GPU_AOT                 "Build AOT for Intel GPU"      OFF)
option(USE_NVIDIA_BACKEND      "Build for NVIDIA backend"     OFF)
option(USE_AMD_BACKEND         "Build for AMD backend"        OFF)
option(DEVICE_TIMER            "Build using Device Timer" OFF)

set(IN_ORDER_QUEUE_DEFAULT OFF)
if (${USE_NVIDIA_BACKEND} OR ${USE_AMD_BACKEND})
    set(IN_ORDER_QUEUE_DEFAULT ON)
endif()
option(IN_ORDER_QUEUE "Use in-order SYCL queue" ${IN_ORDER_QUEUE_DEFAULT})

set(DEF_INTEL_WL_CXX_FLAGS  " -DMKL_ILP64 ")
set(DEF_NVIDIA_WL_CXX_FLAGS " -DUSE_CUBLAS ")
set(DEF_AMD_WL_CXX_FLAGS    " -DUSE_ROCBLAS -D__HIP_PLATFORM_AMD__ ")

set(DEF_INTEL_GENERAL_CXX_FLAGS  " -O3 -fsycl -ffast-math ")
set(DEF_NVIDIA_GENERAL_CXX_FLAGS " -O3 -fsycl -ffast-math ")
set(DEF_AMD_GENERAL_CXX_FLAGS    " -O3 -fsycl -ffast-math ")

if (${IN_ORDER_QUEUE})
    string(APPEND DEF_INTEL_GENERAL_CXX_FLAGS " -DIN_ORDER_QUEUE ")
    string(APPEND DEF_NVIDIA_GENERAL_CXX_FLAGS " -DIN_ORDER_QUEUE ")
    string(APPEND DEF_AMD_GENERAL_CXX_FLAGS " -DIN_ORDER_QUEUE ")
endif()

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




if(NOT DEFINED DNNLROOT AND DEFINED ENV{DNNLROOT})
    set(DNNLROOT "$ENV{DNNLROOT}" CACHE STRING "")
else()
    set(DNNLROOT "${PROJECT_SOURCE_DIR}/.." CACHE STRING "")
endif()

if(USE_AMD_BACKEND)
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
        message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm-4.5.0/hip or the variable HIP_PATH is set to point to the right location.")
    endif()
    set(HIP_SEPARABLE_COMPILATION ON)
endif()

set(SOURCES 
    basic-dl/softmax_layer.cpp
    basic-dl/lnorm_layer.cpp
    ../common/timing.cpp
    ../common/mlp.cpp
    ../common/basic_mem_ops.cpp
    ../common/cait/cait_class_attn_head.cpp
    ../common/cait/cait_multi_headed_class_attn.cpp
    ../common/cait/cait_self_attn_stage.cpp
    ../common/cait/cait.cpp
    ../common/vit/vit_multi_headed_self_attn.cpp
    ../common/vit/vit_self_attn_head.cpp
    ../common/vit/vit_transformer_encoder.cpp
    ../common/vit/vit_transformer.cpp
    ../common/vit/vit.cpp

    ../common/workload_params.cpp
    #handle.cpp
    upsample.cpp
    main.sycl.cpp

    ${CMAKE_SOURCE_DIR}/../../infrastructure/SYCL.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/CommandLineParser.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/TestBenchBase.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/Utilities.cpp
)

if(USE_AMD_BACKEND)
    set(MY_TARGET_NAME ${PROJECT_NAME})
    set(MY_HIPCC_OPTIONS)
    set(MY_NVCC_OPTIONS)
    set(CMAKE_HIP_ARCHITECTURES OFF)
endif()

include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/../common
                    ${CMAKE_SOURCE_DIR}/../../infrastructure
                    #${DNNLROOT}/include
                    ${CUDA_INCLUDE_DIRS}
                    ${CUDA_TOOLKIT_INCLUDE}

                    # ${HIP_PATH}/../include
                    # ${HIP_PATH}/../include/hip
                    # ${HIP_PATH}/../llvm/bin/../lib/clang/14.0.0
                    # ${HIP_PATH}/../hsa/include
                    # ${HIP_PATH}/../miopen/include/miopen             
)

#link_directories(${DNNLROOT}/lib)

message(STATUS "DNNLROOT set to: ${DNNLROOT}")

# Use either default or user defined CXX flags
# -DCMAKE_CXX_FLAGS=" -blah -blah " overrides the default flags




add_compile_options(-DRUN_ON_GPU)
#add_compile_options(-DKERNEL_USE_PROFILE "0")
if(DEVICE_TIMER)
    message(STATUS "Enabling Device Timer")
    add_compile_options(-DDEVICE_TIMER)
endif()

add_executable(${PROJECT_NAME} ${SOURCES})
if(USE_NVIDIA_BACKEND)
    target_link_libraries(${PROJECT_NAME} -lcublas -lcudart -lcuda -lcudnn)
elseif(USE_AMD_BACKEND)
    target_link_libraries(${PROJECT_NAME} -lamdhip64 -lrocblas -lMIOpen)
else()
    message(STATUS "Enabling INTEL backend")
    link_directories(${MKLROOT}/lib/intel64)
    target_link_libraries(${PROJECT_NAME} dnnl mkl_sycl mkl_intel_ilp64 mkl_tbb_thread mkl_core pthread dl m)
endif()

# JIT compilation 
if(GPU_AOT) 
    message(STATUS "Enabling INTEL backend")
    set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}")    
    if( (${GPU_AOT} STREQUAL "pvc") OR (${GPU_AOT} STREQUAL "PVC") )
        message(STATUS "Enabling Intel GPU AOT compilation for ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_gen -Xs \"-device 0x0bd5 -revision_id 3\" ") 
    else()
        message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
    endif()
elseif(USE_NVIDIA_BACKEND)    
    message(STATUS "Enabling NVIDIA backend")
    set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvidia_gpu_sm_${USE_SM}")
elseif(USE_AMD_BACKEND)
    message(STATUS "Enabling AMD backend")
    message(STATUS "Enabling AMD HIP backend for ${USE_AMD_ARCH} AMD architecture")
    set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${USE_AMD_ARCH}  ")
else()
    # JIT case
    message(STATUS "Enabling INTEL backend")
    set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}")
endif()

# Output the compiler flags that were constructed for visual inspection
message(STATUS "Compilation flags set to: ${CMAKE_CXX_FLAGS}")

#link_libraries(stdc++fs dnnl mkl_sycl)
link_libraries(stdc++fs)
#add_executable(${PROJECT_NAME} ${SOURCES})
#target_link_libraries(${PROJECT_NAME} sycl stdc++fs dnnl mkl_sycl mkl_intel_ilp64 mkl_tbb_thread mkl_core)
#target_link_libraries(${PROJECT_NAME} sycl stdc++fs dnnl)
target_link_libraries(${PROJECT_NAME} sycl stdc++fs)
