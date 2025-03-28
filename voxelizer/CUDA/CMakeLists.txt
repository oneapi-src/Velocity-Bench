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

PROJECT(Voxelizer C CXX)

option(USE_SM "Specifies which streaming multiprocessor architecture to use")
option(Trimesh2_LINK_DIR "Path to Trimesh2 library dir")
option(Trimesh2_INCLUDE_DIR "Path to Trimesh2 includes")

set(DEF_WL_CXX_FLAGS " ")
set(DEF_GENERAL_CXX_FLAGS " -O2 ")
set(DEF_COMBINED_CXX_FLAGS "${DEF_GENERAL_CXX_FLAGS} ${DEF_WL_CXX_FLAGS}")

include_directories(
  ${Trimesh2_INCLUDE_DIR}
)

FIND_PACKAGE(glm CONFIG REQUIRED)
message(STATUS "glm library status:")
message(STATUS "    version: ${glm_VERSION}")
message(STATUS "    libraries: ${GLM_LIBRARIES}")
message(STATUS "    include path: ${GLM_INCLUDE_DIRS}")

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

SET(CUDA_VOXELIZER_EXECUTABLE voxelizer_cuda)

# SET(Trimesh2_INCLUDE_DIR CACHE PATH "Path to Trimesh2 includes")
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

SET(CUDA_VOXELIZER_SRCS
  ./src/main.cpp
  ./src/util_cuda.cpp
  ./src/util_io.cpp
  ./src/cpu_voxelizer.cpp
)
SET(CUDA_VOXELIZER_SRCS_CU
  ./src/voxelize.cu
  ./src/thrust_operations.cu
  ./src/voxelize_solid.cu
)

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

# set(CUDA_SEPARABLE_COMPILATION ON)
message(STATUS "CXX  Compilation flags to: ${CMAKE_CXX_FLAGS}")

CUDA_ADD_EXECUTABLE(
  ${CUDA_VOXELIZER_EXECUTABLE}
  ${CUDA_VOXELIZER_SRCS}
  ${CUDA_VOXELIZER_SRCS_CU}
  OPTIONS -arch=sm_${USE_SM})

TARGET_COMPILE_FEATURES(${CUDA_VOXELIZER_EXECUTABLE} PRIVATE cxx_std_17)
TARGET_INCLUDE_DIRECTORIES(${CUDA_VOXELIZER_EXECUTABLE} PRIVATE ${Trimesh2_INCLUDE_DIR} ${GLM_INCLUDE_DIRS}) # TARGET_LINK_LIBRARIES(${CUDA_VOXELIZER_EXECUTABLE} PRIVATE ${Trimesh2_LIBRARY} PRIVATE CUDA::cudart PRIVATE glm::glm)
TARGET_LINK_LIBRARIES(${CUDA_VOXELIZER_EXECUTABLE} ${Trimesh2_LIBRARY} ${CUDA_cudadevrt_LIBRARY} glm::glm OpenMP::OpenMP_CXX)
