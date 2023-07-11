 # Modifications Copyright (C) 2023 Intel Corporation
 #
 # This Program is subject to the terms of the GNU General Public License v3.0 only.​
 # If a copy of the license was not distributed with this file, ​
 # you can obtain one at https://spdx.org/licenses/GPL-3.0-only.html
 #​
 #
 # SPDX-License-Identifier: GPL-3.0-only
 #

cmake_minimum_required(VERSION 3.10)

project(SeisAcoMod2D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MPI_HOME)
    message(STATUS "Using provided MPI_HOME=${MPI_HOME}")
else()
    set(MPI_HOME "/opt/intel/oneapi/mpi/latest")
    message(STATUS "Using default MPI_HOME=${MPI_HOME}")
endif()

option(USE_SM "Build for specific SM" OFF)

# CMAKE_CXX_FLAGS
if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -O3 -std=c++11 -fopenmp ")
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

find_package(MPI REQUIRED)
# message(STATUS "MPI_FOUND: ${MPI_FOUND}")
# message(STATUS "MPI_VERSION: ${MPI_VERSION}")
# message(STATUS "MPIEXEC: ${MPIEXEC}")
# message(STATUS "MPIEXEC_EXECUTABLE: ${MPIEXEC_EXECUTABLE}")
# message(STATUS "MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
# message(STATUS "MPIEXEC_NUMPROC_FLAG: ${MPIEXEC_NUMPROC_FLAG}")
# message(STATUS "MPIEXEC_MAX_NUMPROCS: ${MPIEXEC_MAX_NUMPROCS}")
# message(STATUS "MPIEXEC_PREFLAGS: ${MPIEXEC_PREFLAGS}")
# message(STATUS "MPIEXEC_POSTFLAGS: ${MPIEXEC_POSTFLAGS}")
# message(STATUS "MPI_CXX_COMPILE_FLAGS: ${MPI_CXX_COMPILE_FLAGS}")
# message(STATUS "MPI_CXX_COMPILE_OPTIONS: ${MPI_CXX_COMPILE_OPTIONS}")
# message(STATUS "MPI_CXX_INCLUDE_PATH: ${MPI_CXX_INCLUDE_PATH}")
# message(STATUS "MPI_CXX_LINK_FLAGS: ${MPI_CXX_LINK_FLAGS}")
# message(STATUS "MPI_CXX_LIBRARIES: ${MPI_CXX_LIBRARIES}")

set(SOURCES
    ${CMAKE_SOURCE_DIR}/../common/main.cpp
    ${CMAKE_SOURCE_DIR}/../common/json_parser.cpp
    ${CMAKE_SOURCE_DIR}/../common/modelling_worker.cpp
    ${CMAKE_SOURCE_DIR}/../common/modelling_master.cpp
    ${CMAKE_SOURCE_DIR}/../common/read_receiver.cpp
    ${CMAKE_SOURCE_DIR}/../common/create_geometry.cpp
    ${CMAKE_SOURCE_DIR}/../common/workload.cpp
    ${CMAKE_SOURCE_DIR}/../common/model.cpp
    ${CMAKE_SOURCE_DIR}/../common/check_stability.cpp

    ${CMAKE_SOURCE_DIR}/src/gpu_modelling_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/gpumodelling.cu
)

include_directories(
    ${CMAKE_SOURCE_DIR}/../common/
    ${CMAKE_SOURCE_DIR}/src/
    ${MPI_HOME}/include/
    ${MPI_CXX_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
)

cuda_add_executable(SeisAcoMod2D ${SOURCES})

target_link_libraries(SeisAcoMod2D -L${MPI_HOME}/lib -L${MPI_CXX_LIBRARIES} -L${CUDA_LIBRARIES})
