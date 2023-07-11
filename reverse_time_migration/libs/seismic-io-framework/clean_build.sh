#!/bin/bash

verbose=
serial=-j

while getopts "vsh" opt; do
  case $opt in
  v)
    verbose="VERBOSE=1"
    echo "Using verbose mode"
    ;;
  s)
    serial=
    echo "Using serial mode"
    ;;
  h)
    echo "Usage of $(basename "$0"):"
    echo "	to clean the bin directory then builds the code and run it "
    echo ""
    echo "-v	: to print the output of make in details"
    echo ""
    echo "-s	: to compile serially rather than in parallel"
    echo ""
    exit 1
    ;;
  *)
    echo "Invalid flags entered. run using the -h flag for help"
    exit 1
    ;;
  esac
done

cd bin/ || exit
make clean
make all $serial $verbose
