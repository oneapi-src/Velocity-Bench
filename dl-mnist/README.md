# DL-MNIST

# Description
This workload implements significant aspects of SOTA deep learning architectures (convolutions only and for inference) for image classification using the MNIST dataset. MNIST is one of the oldest and most popular data sets in machine learning.


# DATA
The 4 data files for the DL-MNIST workload that are needed to run it are: 

|Name                           |Size              |
|-------------------------------|------------------|
|t10k-images.idx3-ubyte         |7840016 B         | 
|t10k-labels.idx1-ubyte         |10008 B           | 
|train-images.idx3-ubyte        |47040016 B        | 
|train-labels.idx1-ubyte        |60008 B           |


The datafiles can be downloaded from (http://yann.lecun.com/exdb/mnist/)



Create a directory called 'datasets' in 'dl-mnist' directory. Copy the files into the 'datasets' directory.

**mkdir** dl-mnist/datasets \
**cp** t10k-images.idx3-ubyte dl-mnist/datasets/. \
**cp** t10k-labels.idx1-ubyte dl-mnist/datasets/. \
**cp** train-images.idx3-ubyte.dat dl-mnist/datasets/. \
**cp** train-labels.idx1-ubyte dl-mnist/datasets/. 





# CUDA (utilizing cuDNN library)
To build the cuda program: 

**cd** dl-mnist/CUDA \
**mkdir** build \
**cd** build \
**cmake** .. \
**make** \
**./dl-mnist-cuda -conv_algo CUDNN_FIND_BEST_ALGO** 

**Note**: There is a parameter for cmake called USE_SM. For A100, it's value is 80, and is the default. So it need not be passed in (as shown above). But for other machine types, this needs to be explicitly passed in. For example, running on a RTX6000, the parameter would be USE_SM=75. The full command would be **cmake -DUSE_SM=75 ..** 




# SYCL (utilizing oneDNN library)
To build the sycl program: 

**cd** dl-mnist/SYCL \
**mkdir** build \
**cd** build \
CC=dpcpp CXX=dpcpp **cmake** -DGPU_AOT=PVC .. \
**make** \
**./dl-mnist-sycl  -conv_algo ONEDNN_AUTO** 




# HIP
To build the sycl program: 

**cd** dl-mnist/HIP \
**mkdir** build \
**cd** build \
CC=hipcc CXX=hipcc **cmake** .. \
**make** \
**./dl-mnist-hip  -conv_algo MIOPEN_FIND_BEST_ALGO** 



# SYCL on NVIDIA 
To build the sycl program, you will need **1)** Open source oneAPI DPC++ compiler setup to run on NVIDIA-BACKEND **2)** cudnn and oneDNN would be required: \
We use the flag -DUSE_SM=80 for A100. Use the correct value for your GPU.

**cd** dl-mnist/SYCL \
**mkdir** build \
**cd** build \
CC=clang CXX=clang++ **cmake** -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=80 -DDNNLROOT=/path/to/oneDNN/install/location .. \
**make** \
**ONEAPI_DEVICE_SELECTOR=cuda:gpu ./dl-mnist-sycl   -conv_algo ONEDNN_AUTO** 



# SYCL on AMD 
To build the sycl program, you will need **1)** Open source oneAPI DPC++ compiler setup to run on AMD-BACKEND **2)** MIOpen and oneDNN would be required: \
We use the flag -DUSE_AMDHIP_BACKEND=gfx90a for MI250. Use the correct value for your GPU.

**cd** dl-mnist/SYCL \
**mkdir** build \
**cd** build \
CC=clang CXX=clang++ **cmake** -DUSE_AMDHIP_BACKEND=gfx90a -DDNNLROOT=/path/to/oneDNN/install/location .. \
**make** \
**ONEAPI_DEVICE_SELECTOR=hip:gpu ./dl-mnist-sycl   -conv_algo ONEDNN_AUTO** 


---------------------------------------------------------------------------------------------------------
# Workload logging/tracing

### DL-MNIST provides three kinds of traces: 

**func**  - All function ENTER/EXIT \
**mem**  - Memory operations \
**conv** - Convolution operations 

All three are optional, and can be enabled by providing a comma separated list of them as options to '**-trace**' paramerer. \
*Example:* \
./dl-mnist-sycl  **-trace** func,mem,conv

### oneDNN and cuDNN logging

DL-MNIST uses oneDNN and cuDNN. They can be enabled by setting certain environment variables described below: \
**oneDNN** - https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html \
**cuDNN**   - https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#api-logging 

