#!/bin/bash

#==============================================================================
# author		      :zeyad-osama
# usage		        :sudo ./<script-name>
# bash_version    :4.4.20(1)-release
#==============================================================================

echo "Boost download script assumes you have sudo access..."

# Save PWD to return back to it at the end.
dir="$(pwd)"

# Create directory for installations.
cd ~ || exit
if [ ! -d "hpclibs" ]; then
  mkdir hpclibs && cd hpclibs || exit
fi

# Download.
wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2
tar --bzip2 -xf boost_1_64_0.tar.bz2
cd boost_1_64_0 || exit
./bootstrap.sh --prefix=/usr/
./b2
sudo ./b2 install

# Return to PWD.
cd "$dir" || exit
