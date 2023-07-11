#!/bin/bash

#==============================================================================
# author		      :zeyad-osama
# usage		        :sudo ./<script-name>
# bash_version    :4.4.20(1)-release
#==============================================================================

echo "VTI BP download script assumes you have sudo access..."

echo "This script will download the BP velocity model, density model and all shot files"
echo "The size of the data to be downloaded is around 8 GB, after extraction it will take 13 GB"

# Save PWD to return back to it at the end.
dir="$(pwd)"

cd ../..

if [ ! -d "data" ]; then
  mkdir data
fi

cd data || exit

if [ ! -d "vti" ]; then
  mkdir vti
fi

cd vti || exit

wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/README_Modification.txt
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/DatasetInformation_And_Disclaimer.txt

if [ ! -d "params" ]; then
  mkdir params
fi

cd params || exit

wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/ModelParams.tar.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/OtherFiles-2.tar.gz

cd .. || exit

if [ ! -d "shots" ]; then
  mkdir shots
fi

cd shots || exit

wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/Anisotropic_FD_Model_Shots_part1.sgy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/Anisotropic_FD_Model_Shots_part2.sgy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/Anisotropic_FD_Model_Shots_part3.sgy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/Anisotropic_FD_Model_Shots_part4.sgy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/Anisotropic_FD_Model_VSP.tar.gz

# Return to PWD.
cd "$dir" || exit
