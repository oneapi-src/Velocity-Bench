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
	PoolURI.cpp PoolURI.h
	PoolClient.h
	PoolManager.h PoolManager.cpp
	testing/SimulateClient.h testing/SimulateClient.cpp
	stratum/EthStratumClient.h stratum/EthStratumClient.cpp
	getwork/EthGetworkClient.h getwork/EthGetworkClient.cpp
)

find_package(OpenSSL REQUIRED)
include_directories(${jsoncpp_DIR}/../../../include)
add_library(poolprotocols ${SOURCES})
target_link_libraries(poolprotocols PRIVATE devcore ethminer-buildinfo ethash::ethash Boost::system JsonCpp::JsonCpp OpenSSL::SSL OpenSSL::Crypto)
target_include_directories(poolprotocols PRIVATE ..)
