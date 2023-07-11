### 
### Modifications Copyright (C) 2023 Intel Corporation
### 
### This Program is subject to the terms of the GNU Lesser General Public License v3.0 or later
### 
### If a copy of the license was not distributed with this file, you can obtain one at 
### https://www.gnu.org/licenses/lgpl-3.0-standalone.html
### 
### SPDX-License-Identifier: LGPL-3.0-or-later
### 
### 

# Set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)

# Project settings.
project(Thoth LANGUAGES CXX VERSION 1.0.0)

# Set C++ standard to C++14 to allow for the advanced features used in code.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Definitions
if ("${DATA_PATH}" STREQUAL "")
    SET(DATA_PATH "./data")
endif ()
if ("${WRITE_PATH}" STREQUAL "")
    SET(WRITE_PATH "./results")
endif ()
if ("${WORKLOAD_PATH}" STREQUAL "")
    SET(WORKLOAD_PATH "./workloads")
endif ()

add_definitions(
        -DDATA_PATH="${PROJECT_SOURCE_DIR}/${DATA_PATH}"
        -DWRITE_PATH="${PROJECT_SOURCE_DIR}/${WRITE_PATH}"
        -DWORKLOAD_PATH="${PROJECT_SOURCE_DIR}/${WORKLOAD_PATH}")

# Adding the headers directories needed for all projects.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/prerequisites)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Adding subdirectories
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

option(BUILD_EXAMPLES "Option to enable building example programs" OFF)
if (BUILD_EXAMPLES)
    message(STATUS "Building Examples")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
endif ()

option(BUILD_TESTS "Option to enable building tests" OFF)
if (BUILD_TESTS)
    message(STATUS "Building Tests")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
    include(CTest)
    add_test(NAME THOTH-TESTS
            COMMAND thoth-tests
            )
endif ()
