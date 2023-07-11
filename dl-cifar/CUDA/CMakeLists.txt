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
project(dl-cifar-cuda LANGUAGES CXX) 
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_SM                  "Specifies which streaming multiprocessor architecture to use"     )
option(DEVICE_TIMER            "Build using Device Timer" OFF)

set(SOURCES 
    
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
    main.cuda.cpp
        
    ${CMAKE_SOURCE_DIR}/../../infrastructure/CommandLineParser.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/TestBenchBase.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/Utilities.cpp

)

include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/../common
                    ${CMAKE_SOURCE_DIR}/../../infrastructure
                    ${CUDA_INCLUDE_DIRS}
                    ${CUDA_TOOLKIT_INCLUDE}
)
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT CXX compilation flags for the application")
    string(APPEND CMAKE_CXX_FLAGS " -O2 -g")
else()
    message(STATUS "OVERRIDING CXX compilation flags")
endif()

add_compile_options(-DRUN_ON_GPU)
add_compile_options(-DKERNEL_USE_PROFILE)
add_compile_options(-DUSE_CUDA)
add_compile_options(-DENABLE_CUDA_LOGGING)
if(DEVICE_TIMER)
    message(STATUS "Enabling Device Timer")
    add_compile_options(-DDEVICE_TIMER)
endif()

set(CUDA_NVCC_FLAGS -std=c++17)

if(USE_SM)
    message(STATUS "Building for SM_${USE_SM} compatibility")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${USE_SM},code=sm_${USE_SM} )
else()
    message("-- Building for SM_80 compatibility (DEFAULT)")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80 )
endif()


set(CUDA_SEPARABLE_COMPILATION ON)
message(STATUS "CXX  Compilation flags to: ${CMAKE_CXX_FLAGS}")
link_libraries(stdc++fs cublas cudnn)
cuda_add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})