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

set(STBX-SOURCES

        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/NormalAgent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/StaticServerlessAgent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/StaticServerAgent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/DynamicServerlessAgent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/DynamicServerAgent.cpp

        ${STBX-SOURCES}
        PARENT_SCOPE
        )