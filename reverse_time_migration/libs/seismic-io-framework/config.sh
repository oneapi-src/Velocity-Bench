#!/bin/bash

PROJECT_SOURCE_DIR=$(dirname "$0")
echo "working on directory $PROJECT_SOURCE_DIR"

USE_OpenCV="ON"
USE_INTEL="NO"

rm -rf ${PROJECT_SOURCE_DIR}/bin
mkdir bin/

cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DUSE_OpenCV=$USE_OpenCV \
  -DUSE_INTEL=$USE_INTEL \
  -H"${PROJECT_SOURCE_DIR}" \
  -B"${PROJECT_SOURCE_DIR}"/bin
