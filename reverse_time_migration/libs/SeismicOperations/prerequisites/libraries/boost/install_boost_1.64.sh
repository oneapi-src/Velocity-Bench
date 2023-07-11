#!/bin/bash

echo "Boost download script assumes you have sudo access..."

wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2
tar --bzip2 -xf boost_1_64_0.tar.bz2
cd boost_1_64_0 || exit
./bootstrap.sh --prefix=/usr/
./b2
sudo ./b2 install
