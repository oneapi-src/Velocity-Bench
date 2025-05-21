
# MIT License

# Copyright (c) 2015 University of West Bohemia

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



# MIT License

# Modifications Copyright (C) 2023 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-License-Identifier: MIT License






cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

project(svm_hip LANGUAGES CXX)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -O3 -Wextra -D__HIP_PLATFORM_AMD__")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-I/opt/rocm/hipblas/include")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-L/opt/rocm/hipblas/lib/ -lhipblas")

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm-4.5.0/hip or the variable HIP_PATH is set to point to the right location.")
endif()
set(HIP_SEPARABLE_COMPILATION ON)

set(MY_SOURCE_FILES
    cuSVM/cuSVMSolver.cpp
    cuSVM/cuSVM_wrapper.cpp
    libSVM/libSVM_utils.cpp
    libSVM/svm.cpp
    libSVM/libSVM_wrapper.cpp
    debug.cpp
    svm_template.cpp
    svm-train.cpp
    utils.cpp)
include_directories(${CMAKE_SOURCE_DIR}/. ${CMAKE_SOURCE_DIR}/cuSVM ${CMAKE_SOURCE_DIR}/libSVM)

set(MY_TARGET_NAME ${PROJECT_NAME})
set(MY_HIPCC_OPTIONS)
set(MY_NVCC_OPTIONS)
set(CMAKE_HIP_ARCHITECTURES OFF)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../data/a9a
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set_source_files_properties(${MY_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${MY_TARGET_NAME} ${MY_SOURCE_FILES} HIPCC_OPTIONS ${MY_HIPCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})
