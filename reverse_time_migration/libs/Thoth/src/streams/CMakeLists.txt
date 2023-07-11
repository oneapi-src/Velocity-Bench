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

set(SOURCES

        # PRIMITIVE
        ${CMAKE_CURRENT_SOURCE_DIR}/primitive/Reader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitive/Writer.cpp

        # READERS
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/readers/BinaryReader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/readers/CSVReader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/readers/SegyReader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/readers/SUReader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/readers/TextReader.cpp

        # WRITERS
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/writers/BinaryWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/writers/CSVWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/writers/ImageWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/writers/SegyWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/writers/SUWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concrete/writers/TextWriter.cpp

        # HELPERS
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/InStreamHelper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/OutStreamHelper.cpp

        ${SOURCES}
        PARENT_SCOPE
        )