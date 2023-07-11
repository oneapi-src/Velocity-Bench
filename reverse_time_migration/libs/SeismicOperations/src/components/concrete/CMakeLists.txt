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

# TECHNOLOGY
if (USE_DPC)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/oneapi)
elseif (USE_CUDA)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuda)
elseif (USE_HIP)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hip)
endif ()

set(OPERATIONS-SOURCES ${OPERATIONS-SOURCES} PARENT_SCOPE)
