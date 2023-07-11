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
project(dl-cifar-hip LANGUAGES CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DEVICE_TIMER            "Build using Device Timer" OFF)


#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -Wall -O3 -Wextra -D__HIP_PLATFORM_AMD__")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -Wall -O3 -Wextra")

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

set(MY_SOURCE_FILES 
     
    basic-dl/lnorm_layer.cu
    basic-dl/softmax_layer.cpp
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
    upsample.cu    
    main.hip.cpp

    ${CMAKE_SOURCE_DIR}/../../infrastructure/CommandLineParser.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/TestBenchBase.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/Utilities.cpp

)


set(MY_TARGET_NAME ${PROJECT_NAME})
set(MY_HIPCC_OPTIONS)
set(MY_NVCC_OPTIONS)
set(CMAKE_HIP_ARCHITECTURES OFF)

include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/../common
                    ${CMAKE_SOURCE_DIR}/../../infrastructure

                    ${HIP_PATH}/../include
                    ${HIP_PATH}/../include/hip
                    ${HIP_PATH}/../llvm/bin/../lib/clang/14.0.0
                    ${HIP_PATH}/../hsa/include
                    ${HIP_PATH}/../miopen/include/miopen
)


if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT CXX compilation flags for the application")
    string(APPEND CMAKE_CXX_FLAGS " -O2 -g")
else()
    message(STATUS "OVERRIDING CXX compilation flags")
endif()

add_compile_options(-DRUN_ON_GPU)
add_compile_options(-DKERNEL_USE_PROFILE)
add_compile_options(-DUSE_HIP)
add_compile_options(-DENABLE_HIP_LOGGING)
if(DEVICE_TIMER)
    message(STATUS "Enabling Device Timer")
    add_compile_options(-DDEVICE_TIMER)
endif()


message(STATUS "CXX  Compilation flags to: ${CMAKE_CXX_FLAGS}")
link_libraries(stdc++ MIOpen rocblas)
hip_add_executable(${MY_TARGET_NAME} ${MY_SOURCE_FILES} HIPCC_OPTIONS ${MY_HIPCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})
#target_link_libraries(${PROJECT_NAME} ${HIP_LIBRARIES})
