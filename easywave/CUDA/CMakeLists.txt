###  
###  Modifications Copyright (C) 2023 Intel Corporation
###  
###  This Program is subject to the terms of the European Union Public License 1.2
###  
###  If a copy of the license was not distributed with this file, you can obtain one at 
###  https://joinup.ec.europa.eu/sites/default/files/custom-page/attachment/2020-03/EUPL-1.2%20EN.txt
###  
###  SPDX-License-Identifier: EUPL-1.2
###  
###  

cmake_minimum_required(VERSION 3.10)
project(easywave_cuda LANGUAGES CXX) 
find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_KERNEL_PROFILING "Build using kernel profiling"                                  OFF)
option(USE_SM                  "Specifies which streaming multiprocessor architecture to use"     )
option(SHOW_GRID               "Show intermediate grid size during propagation              "  OFF)
option(ENABLE_GPU_TIMINGS      "Show GPU timings at end of execution"                          OFF)

# Host source files for compilation
set(SOURCES
    src/EasyWave.cpp
    ${CMAKE_SOURCE_DIR}/../common/cOgrd.cpp
    ${CMAKE_SOURCE_DIR}/../common/cOkadaEarthquake.cpp
    ${CMAKE_SOURCE_DIR}/../common/cOkadaFault.cpp
    ${CMAKE_SOURCE_DIR}/../common/cSphere.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewGrid.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewOut2D.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewParam.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewPOIs.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewReset.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewSource.cpp
    ${CMAKE_SOURCE_DIR}/../common/ewStep.cpp
    ${CMAKE_SOURCE_DIR}/../common/okada.cpp
    ${CMAKE_SOURCE_DIR}/../common/utilits.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/FileHandler.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/Timer.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/Utilities.cpp
)
include_directories(${CMAKE_SOURCE_DIR}/../common ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/../../infrastructure)

if(ENABLE_KERNEL_PROFILING)
    message(STATUS "Enabling kernel profiling")
    add_compile_options(-DENABLE_KERNEL_PROFILING)
endif()

if(SHOW_GRID)
    message(STATUS "Showing grid size during propagation")
    add_compile_options(-DSHOW_GRID)
endif()

if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT CXX compilation flags for the application")
    string(APPEND CMAKE_CXX_FLAGS " -O3 ")
else()
    message(STATUS "OVERRIDING CXX compilation flags")
endif()
add_compile_options(-DUSE_CUDA)
add_compile_options(-DENABLE_CUDA_LOGGING)

set(CUDA_NVCC_FLAGS -std=c++17)
if(ENABLE_GPU_TIMINGS)
    message(STATUS "GPU Timings will be displayed")
    add_compile_options(-DENABLE_GPU_TIMINGS)
    list(APPEND CUDA_NVCC_FLAGS -DENABLE_GPU_TIMINGS)
endif()

if(USE_SM)
    message(STATUS "Building for SM_${USE_SM} compatibility")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 --use_fast_math -gencode arch=compute_${USE_SM},code=sm_${USE_SM} -DENABLE_CUDA_LOGGING )
endif()



set(CUDA_SEPARABLE_COMPILATION ON)
set(KERNEL_SOURCES "src/ewCudaKernels.cu" "src/ewGpuNode.cu")
message(STATUS "CXX  Compilation flags to: ${CMAKE_CXX_FLAGS}")
include_directories(${CUDA_TOOLKIT_INCLUDE} ${CMAKE_SOURCE_DIR}/../../common ${CMAKE_SOURCE_DIR}/../../infrastructure/ ${CMAKE_SOURCE_DIR}/src/ )
link_libraries(stdc++fs)
cuda_add_executable(${PROJECT_NAME} ${SOURCES} ${KERNEL_SOURCES})
