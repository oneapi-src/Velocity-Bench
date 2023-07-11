cmake_minimum_required(VERSION 3.10)
project(cudaSift C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DEVICE_TIMER            "Build using Device Timer" OFF)
option(USE_SM                  "Specifies which streaming multiprocessor architecture to use"     )

find_package(OpenCV REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  # if (APPLE)
  #   set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
  #   list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
  # else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse2")
    # list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/gcc;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT")
    # list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin") 
    # list(APPEND CUDA_NVCC_FLAGS "-g;-G;-lineinfo;-ccbin;/usr/bin/gcc-8;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT;") 
  # endif()
endif()

set(cuda_sources  
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
)  

set(sources
  ${CMAKE_SOURCE_DIR}/../common/Utility.cpp
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

cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_${USE_SM})
set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
 
install(FILES 
  ${cuda_sources} 
  ${sources}
  cudaSiftD.cu
  CMakeLists.txt
  DESTINATION .
)
install(FILES data/left.pgm data/righ.pgm
  DESTINATION data
)
