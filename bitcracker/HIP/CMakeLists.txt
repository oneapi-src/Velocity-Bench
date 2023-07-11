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

if(NOT DEFINED ROCM_PATH)
    if(NOT DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which HIP has been installed")
    else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${ROCM_PATH}/hip/cmake" ${CMAKE_MODULE_PATH})
set(HIP_INCLUDE_DIRS  "${ROCM_PATH}/include"   ${HIP_INCLUDE_DIRS})
set(HIP_LIBRARIES     "${ROCM_PATH}/lib"       ${HIP_LIBRARIES})

# CMAKE_CXX_FLAGS
if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags for the application")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -O3 -std=c++17 -ffast-math ")
else()
    message(STATUS "OVERRIDING compilation flags")
endif()

find_package(HIP REQUIRED)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_SOURCE_DIR}/src/w_blocks.cpp
    ${CMAKE_SOURCE_DIR}/src/attack.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${HIP_INCLUDE_DIRS}
)

add_executable(bitcracker ${SOURCES})

# target_link_libraries(bitcracker ${HIP_LIBRARIES})
