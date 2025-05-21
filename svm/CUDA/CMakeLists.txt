
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
project(svm_cuda) 
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_SM                  "Specifies which streaming multiprocessor architecture to use"     )

set(SOURCES
    cuSVM/cuSVMSolver.cu  
    cuSVM/cuSVM_wrapper.cpp 
    libSVM/libSVM_utils.cpp  
    libSVM/svm.cpp 
    libSVM/libSVM_wrapper.cpp 
    debug.cpp 
    svm_template.cpp 
    svm-train.cpp utils.cpp)

include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/cuSVM
                    ${CMAKE_SOURCE_DIR}/libSVM)

if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT CXX compilation flags for the application")
    string(APPEND CMAKE_CXX_FLAGS " -O2")
else()
    message(STATUS "OVERRIDING CXX compilation flags")
endif()

#add_compile_options(-DRUN_ON_GPU)
#add_compile_options(-DKERNEL_USE_PROFILE)

set(CMAKE_CUDA_ARCHITECTURES 61) 
#find_package(CUDA REQUIRED)
if(USE_SM)
    message(STATUS "Building for SM_${USE_SM} compatibility")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${USE_SM},code=sm_${USE_SM} )
else()
    message("-- Building for SM_80 compatibility (DEFAULT)")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75)
endif()


set(CUDA_SEPARABLE_COMPILATION ON)
message(STATUS "CXX  Compilation flags to: ${CMAKE_CXX_FLAGS}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../data/a9a
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

include_directories(${CUDA_TOOLKIT_INCLUDE})
set(CUDA_SEPARABLE_COMPILATION ON)
link_libraries(stdc++fs cuda)
cuda_add_executable(${PROJECT_NAME} ${SOURCES})
cuda_add_cublas_to_target(${PROJECT_NAME})

