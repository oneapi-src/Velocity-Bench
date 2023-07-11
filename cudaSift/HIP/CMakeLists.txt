cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED HIP_PATH)
  if(NOT DEFINED ENV{HIP_PATH})
    set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
  else()
    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
  endif()
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

project(cudasift)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -O3 -Wextra  -D__HIP_PLATFORM_AMD__=1")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DEVICE_TIMER "Build using Device Timer" OFF)

find_package(HIP QUIET)

if(HIP_FOUND)
  message(STATUS "Found HIP: " ${HIP_VERSION})
else()
  message(FATAL_ERROR "Could not find HIP!")
endif()

set(HIP_SEPARABLE_COMPILATION ON)

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

if(DEVICE_TIMER)
  message(STATUS "Enabling Device Timer")
  add_compile_options(-DDEVICE_TIMER)
endif()

set(MY_TARGET_NAME ${PROJECT_NAME})
set(MY_HIPCC_OPTIONS)
set(MY_NVCC_OPTIONS)
set(CMAKE_HIP_ARCHITECTURES OFF)
set(CMAKE_NVCC_FLAGS ${CMAKE_NVCC_FLAGS} -std=c++11)

set_source_files_properties(${cuda_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${MY_TARGET_NAME} ${SOURCES} ${MY_HIPCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})
target_link_libraries(cudasift stdc++ stdc++fs ${OpenCV_LIBS})

# SET(CUDA_SEPARABLE_COMPILATION ON)
# hip_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_61)

# cuda_add_executable(l2net l2netD.cu OPTIONS -arch=sm_35)
# set_target_properties(cudasift PROPERTIES
# COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
# )

# target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

# /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS}
# )
# install(FILES
# ${cuda_sources}
# ${sources}
# cudaSiftD.cu
# CMakeLists.txt
# Copyright.txt
# DESTINATION .
# )
# install(FILES data/left.pgm data/righ.pgm
# DESTINATION data
# )
