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

project(sobel_filter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_SM "Build for specific SM" OFF)

# CMAKE_CXX_FLAGS
if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -O3 -ffast-math -m64 -DUSE_CUDA ")
else()
    message(STATUS "OVERRIDING DEFAULT compilation flags")
endif()

# CUDA_NVCC_FLAGS
if(USE_SM)
    message("-- Building for SM_${USE_SM} compatibility")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${USE_SM},code=sm_${USE_SM} )
else()
    message("-- Building for SM_80 compatibility (DEFAULT)")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80 )
endif()

find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED )

set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/sobelfilter.cu
    ${CMAKE_SOURCE_DIR}/../common/common.cpp
    ${CMAKE_SOURCE_DIR}/../infrastructure/Utilities.cpp
    ${CMAKE_SOURCE_DIR}/../infrastructure/CommandLineParser.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}/src/
    ${CMAKE_SOURCE_DIR}/../common/
    ${CMAKE_SOURCE_DIR}/../infrastructure/
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

cuda_add_executable(sobel_filter ${SOURCES})

target_link_libraries(sobel_filter ${CUDA_LIBRARIES} ${OpenCV_LIBS})
