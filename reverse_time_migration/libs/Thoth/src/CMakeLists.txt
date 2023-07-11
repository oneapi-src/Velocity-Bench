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

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/configurations)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/data-units)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/helpers)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/indexers)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lookups)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/properties)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/streams)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils)


add_library(THOTH STATIC ${SOURCES})
target_link_libraries(THOTH ${LIBS})