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
project(dl-mnist-cuda LANGUAGES CXX) 
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_SM                  "Specifies which streaming multiprocessor architecture to use"     )
option(DEVICE_TIMER            "Build using Device Timer" OFF)

set(DEF_WL_CXX_FLAGS           " ")
set(DEF_GENERAL_CXX_FLAGS      " -O2 ")
set(DEF_COMBINED_CXX_FLAGS     "${DEF_GENERAL_CXX_FLAGS} ${DEF_WL_CXX_FLAGS}")

set(SOURCES
    conv_layer.cudnn.cpp
    tensor_mgr.cudnn.cpp
    dl_network_mgr.cudnn.cpp
    main.cudnn.cpp
    ../common/mnist.cpp
    ../common/timing.cpp
    ../common/exec_policies.cpp
    ../common/utils.cpp
    ../common/workload_params.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/CommandLineParser.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/TestBenchBase.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/Utilities.cpp
)
include_directories(
    ${CMAKE_SOURCE_DIR} 
    ${CMAKE_SOURCE_DIR}/../common
    ${CMAKE_SOURCE_DIR}/../../infrastructure
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_INCLUDE})


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

add_compile_options(-DRUN_ON_GPU)
add_compile_options(-DKERNEL_USE_PROFILE)
add_compile_options(-DUSE_CUDA)
add_compile_options(-DENABLE_CUDA_LOGGING)
if(DEVICE_TIMER)
    message(STATUS "Enabling Device Timer")
    add_compile_options(-DDEVICE_TIMER)
endif()

if(USE_SM)
    message(STATUS "Building for SM_${USE_SM} compatibility")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${USE_SM},code=sm_${USE_SM} )
else()
    message("-- Building for SM_80 compatibility (DEFAULT)")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80 )
endif()


set(CUDA_SEPARABLE_COMPILATION ON)
message(STATUS "CXX  Compilation flags to: ${CMAKE_CXX_FLAGS}")
link_libraries(stdc++fs cudnn)
cuda_add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
