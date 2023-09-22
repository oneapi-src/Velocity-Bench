### 
### Modifications Copyright (C) 2023 Intel Corporation
### 
### This Program is subject to the terms of the GNU Lesser General Public License v3.0 or later
### 
### If a copy of the license was not distributed with this file, you can obtain one at 
### https://www.gnu.org/licenses/lgpl-3.0-standalone.html
### 
### SPDX-License-Identifier: LGPL-3.0-or-later
### 
### 

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

project(segy LANGUAGES CXX)

add_library(openCV-vis
        STATIC
        visualization.h visualization.cpp
        )
if (USE_OpenCV)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif ()
target_compile_definitions(openCV-vis PUBLIC $<$<BOOL:${USE_OpenCV}>:USE_OpenCV>)

if (USE_OpenCV)
    if ( (${USE_INTEL} STREQUAL "YES") AND (${USE_NVIDIA_BACKEND} STREQUAL "NO") AND (${USE_AMD_BACKEND} STREQUAL "") )
        target_link_libraries(openCV-vis ${OpenCV_LIBS} "-no-ipo")

    else ()
        target_link_libraries(openCV-vis ${OpenCV_LIBS})
    endif ()
endif ()



