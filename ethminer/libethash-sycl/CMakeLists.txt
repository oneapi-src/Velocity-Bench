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

#find_package(DPCPP REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT MSVC)
	list(APPEND DPCPP_FLAGS "--disable-warnings")
endif()

file(GLOB sources "*.cpp" )
file(GLOB headers "*.h" "hpp" )

add_library(ethash-sycl STATIC ${sources} ${headers})
target_link_libraries(ethash-sycl ethcore)# ethash::ethash Boost::thread)
include_directories(..)

