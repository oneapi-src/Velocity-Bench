#!/bin/bash

#==============================================================================
# author		      :zeyad-osama
# usage		        :sudo ./<script-name>
# bash_version    :4.4.20(1)-release
#==============================================================================

echo "Isotropic BP download script assumes you have sudo access..."

echo "This script will download the BP velocity model, density model and all shot files"
echo "The size of the data to be downloaded is around 8 GB, after extraction it will take 13 GB"

# Save PWD to return back to it at the end.
dir="$(pwd)"

cd ../..

if [ ! -d "data" ]; then
  mkdir data
fi

cd data || exit

if [ ! -d "iso" ]; then
  mkdir iso
fi

cd iso || exit


if [ ! -d "params" ]; then
  mkdir params
fi

cd params || exit

wget http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/density_z6.25m_x12.5m.segy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/vel_z6.25m_x12.5m_exact.segy.gz

gunzip density_z6.25m_x12.5m.segy.gz
gunzip vel_z6.25m_x12.5m_exact.segy.gz

cd .. || exit

if [ ! -d "shots" ]; then
  mkdir shots
fi

cd shots || exit

wget http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/shots0001_0200.segy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/shots0201_0400.segy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/shots0401_0600.segy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/shots0601_0800.segy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/shots0801_1000.segy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/shots1001_1200.segy.gz
wget http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/shots1201_1348.segy.gz

gunzip shots0001_0200.segy.gz
gunzip shots0201_0400.segy.gz
gunzip shots0401_0600.segy.gz
gunzip shots0601_0800.segy.gz
gunzip shots0801_1000.segy.gz
gunzip shots1001_1200.segy.gz
gunzip shots1201_1348.segy.gz

# Return to PWD.
cd "$dir" || exit
