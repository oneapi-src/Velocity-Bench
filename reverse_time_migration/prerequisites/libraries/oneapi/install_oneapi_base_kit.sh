#!/bin/bash

#==============================================================================
# author		      :zeyad-osama
# usage		        :sudo ./<script-name>
# bash_version    :4.4.20(1)-release
#==============================================================================

echo "OneAPI Base Kit download script assumes you have sudo access..."

# Save PWD to return back to it at the end.
dir="$(pwd)"

# Create directory for installations.
cd ~ || exit
if [ ! -d "hpclibs" ]; then
  mkdir hpclibs && cd hpclibs || exit
fi

# Download.
sudo wget https://registrationcenter-download.intel.com/akdlm/irc_nas/17431/l_BaseKit_p_2021.1.0.2659_offline.sh
sudo bash  l_BaseKit_p_2021.1.0.2659_offline.sh

# Return to PWD.
cd "$dir" || exit
