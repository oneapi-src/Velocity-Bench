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

# sets a minimum required version for CMake. A fatal error
#will be issued if a version of CMake lower than that is used
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

#declares the name of the project (recipe-01) and the
#supported language (CXX stands for C++):

project(Seismic-IO LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 17)

add_subdirectory(./Segy)
add_subdirectory(./visualization)


add_library(
        SEISMIC-IO
        STATIC

        datatypes.cpp
        datatypes.h
)

add_executable(datatypes datatypes.cpp)
