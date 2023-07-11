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

set(OPERATIONS-SOURCES

        ${CMAKE_CURRENT_SOURCE_DIR}/filters/noise_filtering.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/interpolation/Interpolator.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/io/location_comparator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/io/read_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/io/write_utils.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/sampling/Sampler.cpp

        ${OPERATIONS-SOURCES}
        PARENT_SCOPE
        )

set(OPERATIONS-LIBS

        MEMORY-MANAGER

        ${OPERATIONS-LIBS}
        PARENT_SCOPE
        )

add_subdirectory(compressor)
