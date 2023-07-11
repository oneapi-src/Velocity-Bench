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

add_definitions(-DTEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test-data/")
add_definitions(-DTEST_RESULTS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test-results/")

set(TESTFILES test_main.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test-utils/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test-utils/src)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/data-units)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/properties)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/streams)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils)

enable_testing()
add_executable(thoth-tests ${TESTFILES})
target_link_libraries(thoth-tests THOTH)
