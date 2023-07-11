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

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/agents)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/generators)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/parsers)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/writers)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils)

if (USE_MPI)
    add_definitions(-DUSING_MPI)
    set(STBX-LIBS SEISMIC-OPERATIONS ${MPI_LIBRARIES})
else ()
    set(STBX-LIBS SEISMIC-OPERATIONS)
endif ()


set(STBX-LIBS

        SEISMIC-OPERATIONS
        SEGY-TOOLS
        SEISMIC-IO
        openCV-vis

        ${STBX-LIBS}
        )

add_library(SEISMIC-TOOLBOX STATIC ${STBX-SOURCES})
target_link_libraries(SEISMIC-TOOLBOX ${STBX-LIBS})

if (MPI_COMPILE_FLAGS)
    set_target_properties(SEISMIC-TOOLBOX PROPERTIES
            COMPILE_FLAGS ${MPI_COMPILE_FLAGS})
endif ()

if (MPI_LINK_FLAGS)
    set_target_properties(SEISMIC-TOOLBOX PROPERTIES
            LINK_FLAGS ${MPI_LINK_FLAGS})
endif ()