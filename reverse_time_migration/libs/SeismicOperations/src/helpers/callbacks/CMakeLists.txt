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

        ${CMAKE_CURRENT_SOURCE_DIR}/primitive/CallbackCollection.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/interface/WriterCallback.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/BinaryWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/CSVWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/ImageWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/NormWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/SegyWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/SUWriter.cpp

        ${OPERATIONS-SOURCES}
        PARENT_SCOPE
        )