# Copyright (c) 2017, Ruslan Baratov
# All rights reserved.

if(TARGET CURL::libcurl)
  return()
endif()

find_package(CURL MODULE REQUIRED)

add_library(CURL::libcurl UNKNOWN IMPORTED)

set_target_properties(
    CURL::libcurl
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CURL_INCLUDE_DIRS}"
    IMPORTED_LOCATION "${CURL_LIBRARIES}"
)
