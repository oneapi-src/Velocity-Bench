#!/bin/bash

#==============================================================================
# author		      :zeyad-osama
# usage		        :sudo ./<script-name>
# bash_version    :4.4.20(1)-release
#==============================================================================

echo "OpenCV download script assumes you have sudo access..."

sudo apt update
sudo apt install libopencv-dev python3-opencv
python3 -c "import cv2; print(cv2.__version__)"
