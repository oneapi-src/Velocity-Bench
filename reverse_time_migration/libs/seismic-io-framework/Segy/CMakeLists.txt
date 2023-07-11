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

#set(CMAKE_CXX_STANDARD 17)

project(segy LANGUAGES CXX)

list(
        APPEND
        _sources
        segyelement.h
        swapbyte.h
        swapbyte.cpp
        suheaders.h
        suheaders.cpp
        segy_helpers.h
        segy_helpers.cpp
        susegy.h
        susegy.cpp
        segy_io_manager.h
        segy_io_manager.cpp
)


add_library(SEGY-TOOLS STATIC ${_sources})


add_executable(segy_test test_segy.cpp)

target_link_libraries(segy_test SEGY-TOOLS)





