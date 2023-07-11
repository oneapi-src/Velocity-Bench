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

cmake_policy(SET CMP0015 NEW)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE .. ${CMAKE_SOURCE_DIR}/CLI11/include)

set(EXECUTABLE ethminer)

file(GLOB HEADERS "*.h")

add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS})
if(MSVC)
	target_sources(${EXECUTABLE} PRIVATE ethminer.rc)
endif()

target_link_libraries(ethminer PRIVATE ethcore poolprotocols devcore ethminer-buildinfo JsonCpp::JsonCpp Boost::system Boost::thread)

if(ETHDBUS)
	find_package(PkgConfig)
	set( ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/lib/x86_64-linux-gnu/pkgconfig" )
	pkg_check_modules(DBUS dbus-1)
	include_directories(${DBUS_INCLUDE_DIRS})
	link_directories(${DBUS_LIBRARY_DIRS})
	target_link_libraries(ethminer PRIVATE ${DBUS_LIBRARIES})
endif()

if(APICORE)
	target_link_libraries(ethminer PRIVATE apicore)
endif()

include(GNUInstallDirs)
install(TARGETS ethminer DESTINATION ${CMAKE_INSTALL_BINDIR})
if(MSVC)
    install(FILES $<TARGET_PDB_FILE:ethminer> DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
endif()

