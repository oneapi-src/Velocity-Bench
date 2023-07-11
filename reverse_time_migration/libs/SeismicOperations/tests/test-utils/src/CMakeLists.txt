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

set(OPERATIONS-TESTFILES

        # DATA GENERATORS
        ${CMAKE_CURRENT_SOURCE_DIR}/dummy-data-generators/DummyGridBoxGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dummy-data-generators/DummyConfigurationMapGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dummy-data-generators/DummyParametersGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dummy-data-generators/DummyModelGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dummy-data-generators/DummyTraceGenerator.cpp

        # NORM CALCULATOR
        ${CMAKE_CURRENT_SOURCE_DIR}/NumberHelpers.cpp

        # ENVIRONMENT HANDLERS
        ${CMAKE_CURRENT_SOURCE_DIR}/EnvironmentHandler.cpp

        ${OPERATIONS-TESTFILES}
        PARENT_SCOPE
        )