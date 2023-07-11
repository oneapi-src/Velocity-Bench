# Prerequisites

This folder includes all the prerequisites needed for developing in seismic toolbox.
All scripts included assumes you have ```sudo``` access. To install and download everything you can easily run the ```setup.sh``` script

## Libraries
### Boost
Folder containing the a [script](libraries/boost/install_boost_1.64.sh) for downloading and installing ```boost``` library.

### Catch2
Folder containing the ```catch.hpp``` header needed to use the Catch2 testing framework used for all the tests of the system.

### OpenCV
Folder containing the a script for downloading and installing ```OpenCV``` library.
* [```apt``` version](libraries/opencv/install_opencv_apt.sh)
* [```git``` version](libraries/opencv/install_opencv_git.sh)

### ZFP Compression
Folder containing the script for downloading and installing ```zfp``` library.
* [```zfp``` library](libraries/zfp/install_zfp.sh) 

## Data
Folder containing the scripts for downloading 2004 BP model (refer [here](https://wiki.seg.org/wiki/2004_BP_velocity_estimation_benchmark_model) for more information)
* [Minimum version (~8GB download)](data-download/download_bp_data_iso.sh)
* [Normal version (~1GB download)](data-download/download_bp_data_iso_minimal.sh)
