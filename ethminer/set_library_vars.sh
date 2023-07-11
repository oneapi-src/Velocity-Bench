#!/bin/bash

# $1 is the root directory path of all the libraries

CheckDirectoryExists() {
    if [ ! -d $1 ]
    then
        echo "Directory `realpath $1` does not exist!"
        exit 1
    fi
}

echo "$1"

export ethash_DIR=${1}/ethash/0.4.3
CheckDirectoryExists $ethash_DIR

export Boost_DIR=${1}/boost/1.73.0
CheckDirectoryExists $Boost_DIR

export OPENSSL_ROOT_DIR=${1}/openssl/1.1.1f
CheckDirectoryExists $OPENSLL_ROOT_DIR

export jsoncpp_DIR=${1}/json/1.9.5
CheckDirectoryExists $jsoncpp_DIR

echo "Successfully set necessary directories"
