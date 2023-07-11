 # Modifications Copyright (C) 2023 Intel Corporation
 #
 # This program is free software; you can redistribute it and/or modify it
 # under the terms of the GNU General Public License version 2, as published
 # by the Free Software Foundation.
 #
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program; if not, see <http://www.gnu.org/licenses/>.
 #
 #
 # SPDX-License-Identifier: GPL-2.0-only
 #

cmake_minimum_required(VERSION 3.10)

project(bitcracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_SM "Build for specific SM" OFF)

# CMAKE_CXX_FLAGS
if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -O3 -ffast-math ")
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

set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cu
    ${CMAKE_SOURCE_DIR}/src/utils.cu
    ${CMAKE_SOURCE_DIR}/src/w_blocks.cu
    ${CMAKE_SOURCE_DIR}/src/attack.cu
)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

cuda_add_executable(bitcracker ${SOURCES})

target_link_libraries(bitcracker ${CUDA_LIBRARIES})
