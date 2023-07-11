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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
	EthashAux.h EthashAux.cpp
	Farm.cpp Farm.h
	Miner.h Miner.cpp
)

include_directories(BEFORE ..)
include_directories(${jsoncpp_DIR}/../../../include)

add_library(ethcore ${SOURCES})
target_link_libraries(ethcore PUBLIC devcore ethash::ethash PRIVATE hwmon)

if(ETHASHCUDA)
	target_link_libraries(ethcore PUBLIC ethash-cuda)
endif()
if(ETHASHHIP)
	target_link_libraries(ethcore PUBLIC ethash-hip)
endif()
if(ETHASHSYCL)
	target_link_libraries(ethcore PUBLIC ethash-sycl)
endif()
