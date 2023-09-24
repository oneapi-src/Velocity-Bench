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

set(DEF_WL_CXX_FLAGS           " -D__HIP_PLATFORM_AMD__ ")
set(DEF_GENERAL_CXX_FLAGS      " -O3 -std=c++17 -ffast-math ")
set(DEF_COMBINED_CXX_FLAGS     "${DEF_GENERAL_CXX_FLAGS} ${DEF_WL_CXX_FLAGS}")

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
