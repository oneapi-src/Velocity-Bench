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

# Memory Tracker Library

# Set minimum CMake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)

# Project settings.
project(Helpers LANGUAGES CXX VERSION 1.0.0)

# set C++ standard to C++17 to allow for the advanced features used in code.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
