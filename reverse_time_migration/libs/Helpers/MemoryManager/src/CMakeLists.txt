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

set(MEMORY-MANAGER-SOURCES

        # MEMORY TRACKER
        ${CMAKE_CURRENT_SOURCE_DIR}/data-units/mem_list.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/data-units/string_list.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/managers/memory_tracker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/mem_utils.cpp

        # MEMORY TRACKER
        ${CMAKE_CURRENT_SOURCE_DIR}/managers/memory_allocator.cpp
        )

add_library(MEMORY-MANAGER STATIC ${MEMORY-MANAGER-SOURCES})