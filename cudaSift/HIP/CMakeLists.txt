#  Modifications Copyright (C) 2023 Intel Corporation

#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom
#  the Software is furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
#  OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.

#  SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.10)
project(cudasift LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(DEF_WL_CXX_FLAGS " -D__HIP_PLATFORM_AMD__ ")
set(DEF_GENERAL_CXX_FLAGS " -Wall -O3 -Wextra ")
set(DEF_COMBINED_CXX_FLAGS "${DEF_GENERAL_CXX_FLAGS} ${DEF_WL_CXX_FLAGS}")

if(NOT DEFINED ROCM_PATH)
  if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which HIP has been installed")
  else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
  endif()
endif()

set(CMAKE_MODULE_PATH "${ROCM_PATH}/hip/cmake" ${CMAKE_MODULE_PATH})
set(HIP_INCLUDE_DIRS "${ROCM_PATH}/include" ${HIP_INCLUDE_DIRS})
set(HIP_LIBRARIES "${ROCM_PATH}/lib" ${HIP_LIBRARIES})

option(DEVICE_TIMER "Build using Device Timer" OFF)

find_package(HIP REQUIRED)

if(HIP_FOUND)
  message(STATUS "Found HIP: " ${HIP_VERSION})
else()
  message(FATAL_ERROR "Could not find HIP!")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES
  ${CMAKE_SOURCE_DIR}/../common/Utility.cpp
  cudaImage.cpp
  cudaImage.h
  cudaSiftH.cpp
  cudaSiftH.h
  matching.cpp
  cudaSiftD.h
  cudaSift.h
  cudautils.h
  geomFuncs.cpp
  mainSift.cpp
)

include_directories(
  ${CMAKE_SOURCE_DIR}/../common/
  ${CMAKE_CURRENT_SOURCE_DIR}
)

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

message(STATUS "CXX  Compilation flags to: ${CMAKE_CXX_FLAGS}")

if(DEVICE_TIMER)
  message(STATUS "Enabling Device Timer")
  add_compile_options(-DDEVICE_TIMER)
endif()

set(HIP_SEPARABLE_COMPILATION ON)
set(MY_TARGET_NAME ${PROJECT_NAME})
set(MY_HIPCC_OPTIONS)
set(MY_NVCC_OPTIONS)
set(CMAKE_HIP_ARCHITECTURES OFF)

set_source_files_properties(${cuda_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${MY_TARGET_NAME} ${SOURCES} ${MY_HIPCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})
target_link_libraries(cudasift stdc++ stdc++fs ${OpenCV_LIBS})
