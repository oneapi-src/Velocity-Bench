#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

for arg in "$@"; do
  shift
  case "$arg" in
  "--help") set -- "$@" "-h" ;;
  "--release") set -- "$@" "-r" ;;
  "--images") set -- "$@" "-i" ;;
  "--tests") set -- "$@" "-t" ;;
  "--examples") set -- "$@" "-e" ;;
  *) set -- "$@" "$arg" ;;
  esac
done

while getopts ":tevh:" opt; do
  case $opt in
  t) ##### Building tests enabled #####
    echo -e "${GREEN}Building tests enabled${NC}"
    BUILDING_TESTS="ON"
    ;;

  e) ##### Building examples enabled #####
    echo -e "${GREEN}Building examples enabled${NC}"
    BUILDING_EXAMPLES="ON"
    ;;
  v) ##### printing full output of make #####
    echo -e "${YELLOW}printing make with details${NC}"
    VERBOSE=ON
    ;;
  \?) ##### using default settings #####
    echo -e "${RED}Building tests disabled${NC}"
    echo -e "${RED}Building examples disabled${NC}"
    BUILDING_EXAMPLES="OFF"
    BUILDING_TESTS="OFF"
    VERBOSE=OFF
    ;;
  :) ##### Error in an option #####
    echo "Option $OPTARG requires parameter(s)"
    exit 0
    ;;
  h) ##### Prints the help #####
    echo "Usage of $(basename "$0"):"
    echo ""
    printf "%20s %s\n" "-t :" "to enable building tests"
    echo ""
    printf "%20s %s\n" "-e :" "to enable building examples"
    echo ""
    exit 1
    ;;
  esac
done

if [ -z "$BUILDING_TESTS" ]; then
  BUILDING_TESTS="OFF"
  echo -e "${RED}Building tests disabled${NC}"
fi

if [ -z "$BUILDING_EXAMPLES" ]; then
  BUILDING_EXAMPLES="OFF"
  echo -e "${RED}Building examples disabled${NC}"
fi

rm -rf bin/
mkdir bin/

PROJECT_SOURCE_DIR=$(dirname "$0")
cmake -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_TESTS=$BUILDING_TESTS \
  -DBUILD_EXAMPLES=$BUILDING_EXAMPLES \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=$VERBOSE \
  -H"${PROJECT_SOURCE_DIR}" \
  -B"${PROJECT_SOURCE_DIR}/bin"
