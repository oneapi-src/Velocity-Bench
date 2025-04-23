# DL-CIFAR

# Description
CIFAR-10 is one of the most popular data sets in machine learning today. This workload implements two well known deep learning architectures (VIT and CAIT) for image classification using CIFAR-10.

In this workload, the focus is primarily on characterizing peformance of GPU computation and memory operations (required during training), rather than doing full-blown training.


# DATA
The data files for the DL-CIFAR workload that are needed to run it are: 

|Name                           |Size              |
|-------------------------------|------------------|
|data_batch_1.bin               |30730000 B        | 
|data_batch_2.bin               |30730000 B        | 
|data_batch_3.bin               |30730000 B        | 
|data_batch_4.bin               |30730000 B        |
|data_batch_5.bin               |30730000 B        |
|test_batch.bin.bin             |30730000 B        |
|batches.meta.txt               |61 B              |
|readme.html                    |88 B              |


The CIFAR-10 datafiles are widely available on the internet, and can be downloaded (https://www.cs.toronto.edu/~kriz/cifar.html).



Create a directory called 'datasets/cifar-10-batches-bin' in 'dl-cifar' directory. Unzip the downloaded files above into this directory.








# CUDA 
To build the cuda program: 

**cd** dl-cifar/CUDA \
**mkdir** build \
**cd** build \
**cmake** .. \
**make** \
**./dl-cifar-cuda** 

**Note**: There is a parameter for cmake called USE_SM. For A100, it's value is 80, and is the default. So it need not be passed in (as shown above). But for other machine types, this needs to be explicitly passed in. For example, running on a RTX6000, the parameter would be USE_SM=75. The full command would be **cmake -DUSE_SM=75 ..** 




# SYCL
To build the sycl program: 

**cd** dl-cifar/SYCL \
**mkdir** build \
**cd** build \
CC=dpcpp CXX=dpcpp **cmake** -DGPU_AOT=PVC .. \
**make** \
**./dl-cifar_sycl** 



# HIP
To build the sycl program: 

**cd** dl-cifar/HIP \
**mkdir** build \
**cd** build \
CC=hipcc CXX=hipcc **cmake** .. \
**make** \
**./dl-cifar-hip** 


# SYCL on NVIDIA 
To build the sycl program, you will need **1)** Open source oneAPI DPC++ compiler setup to run on NVIDIA-BACKEND **2)** cuBLAS and cudnn are required: \
We use the flag -DUSE_SM=80 for A100. Use the correct value for your GPU.

**cd** dl-cifar/SYCL \
**mkdir** build \
**cd** build \
CC=clang CXX=clang++ **cmake** -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=80 .. \
**make** \
**ONEAPI_DEVICE_SELECTOR=cuda:gpu ./dl-cifar_sycl** 


# SYCL on AMD 
To build the sycl program, you will need **1)** Open source oneAPI DPC++ compiler setup to run on AMD-BACKEND **2)** rocBLAS and MIOpen are required: \
We use the flag -DUSE_AMD_ARCH=gfx90a for MI250. Use the correct value for your GPU.

**cd** dl-cifar/SYCL \
**mkdir** build \
**cd** build \
CC=clang CXX=clang++ **cmake** -DUSE_AMD_BACKEND=YES -DUSE_AMD_ARCH=gfx90a .. \
**make** \
**ONEAPI_DEVICE_SELECTOR=hip:gpu ./dl-cifar_sycl** 

---------------------------------------------------------------------------------------------------------
## In-order queue
The CMake option `-DIN_ORDER_QUEUE` adds the `in_order` property to the SYCL
queue, as well as `discard_events` if available. The default value of this
option is `ON` for NVIDIA and AMD backends, and `OFF` otherwise.

## Workload logging/tracing

**DL-CIFAR provides function tracing:**

**-trace func**  - All function ENTER/EXIT 

*Example:* \
./dl-cifar-sycl  **-trace** func

## Different network size types

**Ability to specify the network size.** \
**-dl_nw_size_type** <network_size>  

network_size can have the following three values:

**LOW_MEM_GPU_SIZE**       - Small network size useful for low mmory systems and for debugging \
**WORKLOAD_DEFAULT_SIZE**  - This is the network size that is the workflow default \
**FULL_SIZE**              - Full network size. This will likely lead to out of memory errors. And hence is not the default for this workload 

Note: This is not mandatory to be explicitly passed in. The workload uses **WORKLOAD_DEFAULT_SIZE** by default, if nothing is passed in.

*Example:* \
./dl-cifar-sycl  **-dl_nw_size_type WORKLOAD_DEFAULT_SIZE** 


