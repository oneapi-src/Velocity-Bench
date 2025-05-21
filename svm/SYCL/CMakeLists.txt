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
option(USE_AMD_BACKEND      "Build for AMD HIP backend"    OFF)

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

include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/libSVM
                    ${CMAKE_SOURCE_DIR}/cuSVM
                    ${CMAKE_SOURCE_DIR}/infrastructure)


# Use either default or user defined CXX flags
# -DCMAKE_CXX_FLAGS=" -blah -blah " overrides the default flags
set(DEF_INTEL_WL_CXX_FLAGS  " -DMKL_ILP64 ")
set(DEF_NVIDIA_WL_CXX_FLAGS " -DUSE_CUBLAS ")
set(DEF_AMD_WL_CXX_FLAGS    " -DUSE_HIPBLAS ")

set(DEF_INTEL_GENERAL_CXX_FLAGS  " -O3 -fsycl ")
set(DEF_NVIDIA_GENERAL_CXX_FLAGS " -O3 -fsycl ")
set(DEF_AMD_GENERAL_CXX_FLAGS    " -O3 -fsycl -D__HIP_PLATFORM_AMD__ ")

if(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "" AND NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(FATAL_ERROR "Both  CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS cannot be passed in together")
elseif("${CMAKE_CXX_FLAGS}" STREQUAL "" AND "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags")
    set(INTEL_GPU_CXX_FLAGS        "${DEF_INTEL_GENERAL_CXX_FLAGS}   ${DEF_INTEL_WL_CXX_FLAGS}")
    set(NVIDIA_GPU_CXX_FLAGS       "${DEF_NVIDIA_GENERAL_CXX_FLAGS}  ${DEF_NVIDIA_WL_CXX_FLAGS}")
    set(AMD_GPU_CXX_FLAGS          "${DEF_AMD_GENERAL_CXX_FLAGS}     ${DEF_AMD_WL_CXX_FLAGS}")
elseif(NOT "${OVERRIDE_GENERAL_CXX_FLAGS}" STREQUAL "")
    message(STATUS "OVERRIDING GENERAL compilation flags")
    set(INTEL_GPU_CXX_FLAGS        "${OVERRIDE_GENERAL_CXX_FLAGS}    ${DEF_INTEL_WL_CXX_FLAGS}")
    set(NVIDIA_GPU_CXX_FLAGS       "${OVERRIDE_GENERAL_CXX_FLAGS}    ${DEF_NVIDIA_WL_CXX_FLAGS}")
    set(AMD_GPU_CXX_FLAGS          "${OVERRIDE_GENERAL_CXX_FLAGS}    ${DEF_AMD_WL_CXX_FLAGS}")
elseif(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "OVERRIDING GENERAL and WORKLOAD SPECIFIC compilation flags")
    set(INTEL_GPU_CXX_FLAGS        "${CMAKE_CXX_FLAGS}")
    set(NVIDIA_GPU_CXX_FLAGS       "${CMAKE_CXX_FLAGS}")
    set(AMD_GPU_CXX_FLAGS          "${CMAKE_CXX_FLAGS}")
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
    link_directories(${MKLROOT}/lib/intel64)
    target_link_libraries(${PROJECT_NAME} mkl_sycl mkl_intel_ilp64 mkl_tbb_thread mkl_core pthread dl m)
endif()

if(GPU_AOT) 
    message(STATUS "Enabling INTEL backend")
    set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}")
    if( (${GPU_AOT} STREQUAL "pvc") OR (${GPU_AOT} STREQUAL "PVC") )
        message(STATUS "Enabling Intel GPU AOT compilation for ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=spir64_gen -Xs \"-device 0x0bd5 -revision_id 3\" ") 
    else()
        message(STATUS "Using custom AOT compilation flag ${GPU_AOT}")
        string(APPEND CMAKE_CXX_FLAGS " ${GPU_AOT} ") # User should be aware of advanced AOT compilation flags
    endif()
elseif(USE_NVIDIA_BACKEND)    
    message(STATUS "Enabling NVIDIA backend")
    set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvidia_gpu_sm_${USE_SM} ") 
elseif(USE_AMDHIP_BACKEND)
    message(STATUS "Enabling AMD HIP backend for ${USE_AMDHIP_BACKEND} AMD architecture")
    set(CMAKE_CXX_FLAGS "${AMD_GPU_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=amd_gpu_gfx${USE_AMD_ARCH}  ")
else()
    # JIT case
    message(STATUS "Enabling INTEL backend")
    set(CMAKE_CXX_FLAGS "${INTEL_GPU_CXX_FLAGS}")
endif()

# Output the compiler flags that were constructed for visual inspection
message(STATUS "Compilation flags set to: ${CMAKE_CXX_FLAGS}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../data/a9a
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

#add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} sycl stdc++fs)
