
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
# */

# /*
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
project(svm_sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)           # SYCL code requires this
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Enable modern C++ standards
set(CMAKE_CXX_EXTENSIONS OFF)        # Use -std, not -gnu


option(GPU_AOT                 "Build AOT for Intel GPU"      OFF)
option(USE_NVIDIA_BACKEND      "Build for NVIDIA backend"     OFF)
option(USE_AMDHIP_BACKEND      "Build for AMD HIP backend"    OFF)

set(SOURCES
    cuSVM/cuSVMSolver.dp.cpp
    cuSVM/cuSVM_wrapper.cpp
    libSVM/libSVM_utils.cpp
    libSVM/svm.cpp
    libSVM/libSVM_wrapper.cpp
    svm_template.cpp
    svm-train.cpp
    utils.cpp
    #infrastructure/CommandLineParser.cpp
    #infrastructure/Utilities.cpp
    #infrastructure/SYCL.cpp
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/libSVM
                    ${CMAKE_SOURCE_DIR}/cuSVM
                    ${CMAKE_SOURCE_DIR}/infrastructure)



# Use either default or user defined CXX flags
# -DCMAKE_CXX_FLAGS=" -blah -blah " overrides the default flags

set(INTEL_GPU_CXX_FLAGS  " -O3 -fsycl")
set(NVIDIA_GPU_CXX_FLAGS " -O3 -fsycl -DUSE_CUBLAS")
set(AMD_GPU_CXX_FLAGS    " -O3 -fsycl -DUSE_HIPBLAS -D__HIP_PLATFORM_AMD__")

set(USE_DEFAULT_FLAGS ON)
if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags")
else()
    message(STATUS "OVERRIDING DEFAULT compilation flags")
    set(USE_DEFAULT_FLAGS OFF)
endif()

#add_compile_options(-DRUN_ON_GPU)
#add_compile_options(-DKERNEL_USE_PROFILE "0")

# JIT compilation

add_executable(${PROJECT_NAME} ${SOURCES})
if(USE_NVIDIA_BACKEND)
    target_link_libraries(${PROJECT_NAME} -lcublas -lcudart -lcuda)
elseif(USE_AMD_BACKEND)
    target_link_libraries(${PROJECT_NAME} -lhipblas)
else()
    message(STATUS "Enabling INTEL backend")
    find_package(oneMKL REQUIRED CONFIG HINTS ${oneMKLROOT})
    message(STATUS "Found oneMKL: ${oneMKL_DIR}")
    target_link_libraries(${PROJECT_NAME} PRIVATE MKL::onemkl)
endif()

if(GPU_AOT)
    message(STATUS "Enabling INTEL backend")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}") # Default flags for Intel backend
    endif()
    message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
    string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
elseif(USE_NVIDIA_BACKEND)
    message(STATUS "Enabling NVIDIA backend")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}") # Default flags for NV backend
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvidia_gpu_sm_${USE_SM}")
elseif(USE_AMD_BACKEND)
    message(STATUS "Enabling AMD HIP backend.")
    if(USE_DEFAULT_FLAGS)
        set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}") # Default flags for AMD backend (gfx908 for MI100)
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amd_gpu_gfx${USE_AMD_ARCH}")
endif()

# Output the compiler flags that were constructed for visual inspection
message(STATUS "Compilation flags set to: ${CMAKE_CXX_FLAGS}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/a9a
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

#add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE sycl stdc++fs)
