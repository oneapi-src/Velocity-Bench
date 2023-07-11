# Copyright (c) 2017, Ruslan Baratov
# All rights reserved.

if(NOT TARGET libjson-rpc-cpp::common)
  unset(_jsonrpccpp_lib)
  unset(_jsonrpccpp_lib CACHE)

  find_library(_jsonrpccpp_lib NAMES jsonrpccpp-common)
  if(NOT _jsonrpccpp_lib)
    message(FATAL_ERROR "Library not found")
  endif()

  add_library(libjson-rpc-cpp::common UNKNOWN IMPORTED)

  set_target_properties(
      libjson-rpc-cpp::common
      PROPERTIES
      IMPORTED_LOCATION "${_jsonrpccpp_lib}"
  )

  unset(_jsonrpccpp_lib)
  unset(_jsonrpccpp_lib CACHE)
endif()

if(NOT TARGET libjson-rpc-cpp::client)
  unset(_jsonrpccpp_lib)
  unset(_jsonrpccpp_lib CACHE)

  find_library(_jsonrpccpp_lib NAMES jsonrpccpp-client)
  if(NOT _jsonrpccpp_lib)
    message(FATAL_ERROR "Library not found")
  endif()

  add_library(libjson-rpc-cpp::client UNKNOWN IMPORTED)

  set_target_properties(
      libjson-rpc-cpp::client
      PROPERTIES
      IMPORTED_LOCATION "${_jsonrpccpp_lib}"
      INTERFACE_LINK_LIBRARIES libjson-rpc-cpp::common
  )

  unset(_jsonrpccpp_lib)
  unset(_jsonrpccpp_lib CACHE)
endif()
