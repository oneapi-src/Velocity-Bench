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

project(hashtable LANGUAGES CXX)

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
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -O3 -std=c++17 -ffast-math -D__HIP_PLATFORM_AMD__ ")
else()
    message(STATUS "OVERRIDING compilation flags")
endif()

find_package(HIP REQUIRED)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/test.cpp
    ${CMAKE_SOURCE_DIR}/src/linearprobing.cpp)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${HIP_INCLUDE_DIRS}
)

add_executable(hashtable ${SOURCES})

target_link_libraries(hashtable -L${HIP_LIBRARIES})
