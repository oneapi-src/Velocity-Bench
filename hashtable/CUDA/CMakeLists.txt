 # Modifications Copyright (C) 2023 Intel Corporation
 #
 # This Program is subject to the terms of The Unlicense.​
 # If a copy of the license was not distributed with this file, ​
 # you can obtain one at https://spdx.org/licenses/Unlicense.html​
 #​
 #
 # SPDX-License-Identifier: Unlicense
 #

cmake_minimum_required(VERSION 3.10)

project(hashtable_cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # use std instead of gnu

option(USE_SM                   "Build for specific SM"         OFF)

# CMAKE_CXX_FLAGS
if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags for the application")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -O3 -Wall -Wextra -Wno-unused-parameter ")
else()
    message(STATUS "OVERRIDING compilation flags")
endif()

add_compile_options(-DUSE_CUDA)

# CUDA_NVCC_FLAGS
if(USE_SM)
    message("-- Building for SM_${USE_SM} compatibility")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${USE_SM},code=sm_${USE_SM} )
else()
    message("-- Building for SM_80 compatibility (DEFAULT)")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80 )
endif()

message("-- Building for CUDA NVCC compiler")
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/test.cpp
    ${CMAKE_SOURCE_DIR}/src/linearprobing.cu
)

include_directories(
    ${CUDA_TOOLKIT_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
    /usr/local/cuda/include
)

cuda_add_executable(${PROJECT_NAME} ${SOURCES})
