### 
### Copyright (C) <2023> Intel Corporation
### 
### This program is free software; you can redistribute it and/or modify it
### under the terms of the GNU General Public License, as published
### by the Free Software Foundation; either version 2 of the License,
### or (at your option) any later version.
###  
### This program is distributed in the hope that it will be useful,
### but WITHOUT ANY WARRANTY; without even the implied warranty of
### MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
### GNU General Public License for more details.
###  
### You should have received a copy of the GNU General Public License
### along with this program; if not, see <http://www.gnu.org/licenses/>.
###  
### 
### SPDX-License-Identifier: GPL-2.0-or-later
### 
### 

set(SOURCES
    wraphelper.cpp wraphelper.h
    wrapnvml.h wrapnvml.cpp
    wrapadl.h wrapadl.cpp
    wrapamdsysfs.h wrapamdsysfs.cpp
)

add_library(hwmon ${SOURCES})
set(HWMON_LINK_LIBRARIES devcore)

if (UNIX)
	list(APPEND HWMON_LINK_LIBRARIES dl)
endif ()

target_link_libraries(hwmon ${HWMON_LINK_LIBRARIES})
target_include_directories(hwmon PRIVATE ..)

if (ETHASHCUDA)
	find_package(CUDA REQUIRED)
	target_include_directories(hwmon PUBLIC ${CUDA_INCLUDE_DIRS})
endif()
