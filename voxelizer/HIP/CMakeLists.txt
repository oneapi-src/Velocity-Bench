# Modifications Copyright (C) 2023 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
# OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

# SPDX-License-Identifier: MIT

CMAKE_MINIMUM_REQUIRED(VERSION 3.20 FATAL_ERROR)
PROJECT(Voxelizer LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
option(Trimesh2_LINK_DIR "Path to Trimesh2 library dir")
option(Trimesh2_INCLUDE_DIR "Path to Trimesh2 includes")
option(GLM_INCLUDE_DIR "Path to GLM includes")

set(DEF_WL_CXX_FLAGS " -D__HIP_PLATFORM_AMD__ ")
set(DEF_GENERAL_CXX_FLAGS " -Wall -O3 -Wextra ")
set(DEF_COMBINED_CXX_FLAGS "${DEF_GENERAL_CXX_FLAGS} ${DEF_WL_CXX_FLAGS}")

include_directories(
  ${Trimesh2_INCLUDE_DIR}
)

# SET(GLM_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../../../dependencies/test/glm/glm)
FIND_PACKAGE(glm REQUIRED)
message(STATUS "glm library status:")
message(STATUS "    version: ${glm_VERSION}")
message(STATUS "    libraries: ${GLM_LIBRARIES}")
message(STATUS "    include path: ${GLM_INCLUDE_DIR}")

if(NOT DEFINED HIP_PATH)
  if(NOT DEFINED ENV{HIP_PATH})
    set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
  else()
    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
  endif()
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
option(WG_SIZE "Specifies the block size")

if("${CMAKE_CXX_FLAGS}" STREQUAL "")
  message(STATUS "Using DEFAULT compilation flags for the application")
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -O3 -std=c++17 -ffast-math -D__HIP_PLATFORM_AMD__ ")
else()
  message(STATUS "OVERRIDING compilation flags")
endif()

# set(HIP_PATH /opt/rocm-5.4.3/hip)
find_package(HIP REQUIRED)

if(NOT HIP_FOUND)
  message(FATAL_ERROR "HIP compiler and libraries were not found")
else()
  message(STATUS "HIP Compiler found in path " ${HIP_PATH})
endif()

SET(HIP_VOXELIZER_EXECUTABLE voxelizer_hip)

# SET(GLM_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src)
IF(NOT Trimesh2_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "You need to set variable Trimesh2_INCLUDE_DIR")
ENDIF()

FIND_FILE(Trimesh2_TriMesh_h TriMesh.h ${Trimesh2_INCLUDE_DIR})

IF(NOT Trimesh2_TriMesh_h)
  message(FATAL_ERROR "Can't find TriMesh.h in ${Trimesh2_INCLUDE_DIR}")
ENDIF()

MARK_AS_ADVANCED(Trimesh2_TriMesh_h)

# SET(Trimesh2_LINK_DIR CACHE PATH "Path to Trimesh2 library dir.")
IF(NOT Trimesh2_LINK_DIR)
  MESSAGE(FATAL_ERROR "You need to set variable Trimesh2_LINK_DIR")
ENDIF()

IF(NOT EXISTS "${Trimesh2_LINK_DIR}")
  MESSAGE(FATAL_ERROR "Trimesh2 library dir does not exist")
ENDIF()

FIND_LIBRARY(Trimesh2_LIBRARY trimesh ${Trimesh2_LINK_DIR})

IF(NOT Trimesh2_LIBRARY)
  message(SEND_ERROR "Can't find libtrimesh.a in ${Trimesh2_LINK_DIR}")
ENDIF()

MARK_AS_ADVANCED(Trimesh2_LIBRARY)

MESSAGE(STATUS "Found Trimesh2 include: ${Trimesh2_TriMesh_h}")
MESSAGE(STATUS "Found Trimesh2 lib: ${Trimesh2_LIBRARY}")

SET(HIP_VOXELIZER_SRCS
  ./src/main.cpp
  ./src/util_cuda.cpp
  ./src/util_io.cpp
  ./src/cpu_voxelizer.cpp
  ./src/voxelize.cpp
  ./src/thrust_operations.cpp
  ./src/voxelize_solid.cpp
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

set(HIP_SEPARABLE_COMPILATION ON)
set(MY_TARGET_NAME ${PROJECT_NAME})
set(MY_HIPCC_OPTIONS)
set(MY_NVCC_OPTIONS)
set(CMAKE_HIP_ARCHITECTURES OFF)

set_source_files_properties(${HIP_VOXELIZER_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${HIP_VOXELIZER_EXECUTABLE} ${HIP_VOXELIZER_SRCS} ${MY_HIPCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})

# include(${CMAKE_CURRENT_SOURCE_DIR}/src/glm/glm/CMakeLists.txt)
TARGET_COMPILE_FEATURES(${HIP_VOXELIZER_EXECUTABLE} PRIVATE cxx_std_17)
TARGET_INCLUDE_DIRECTORIES(${HIP_VOXELIZER_EXECUTABLE} PRIVATE ${Trimesh2_INCLUDE_DIR} ${GLM_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${HIP_VOXELIZER_EXECUTABLE} ${Trimesh2_LIBRARY} glm::glm)
