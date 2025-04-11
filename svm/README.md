# SVM

## Description
This workload is based off of the following work. \
http://patternsonascreen.net/cusvm.html \
https://www.csie.ntu.edu.tw/~cjlin/libsvm/


## SYCL
To build and run the SYCL version of the workload. \
cd sycl \
mkdir build \
cd build \
SYCL for L0 backend - CC=icpx CXX=icpx cmake -DGPU_AOT={L0 Device flag} ../ \
SYCL for Nvidia backend - CC=clang CXX=clang++ cmake -DUSE_NVIDIA_BACKEND=TRUE -DUSE_SM={architecture SM to use for nvidia hardware.. i.e 80 for a100} ../ \
SYCL for AMD backend - CC=clang CXX=clang++ cmake -DUSE_AMD_BACKEND=TRUE -DUSE_AMD_ARCH={flag for hip i.e 90a for MI250} ../ \ 
make \


### In-order queue
The CMake option `-DIN_ORDER_QUEUE` adds the `in_order` property to the SYCL
queue, as well as `discard_events` if available. The default value of this
option is `ON` for NVIDIA and AMD backends, and `OFF` otherwise.

### Running the workload
./svm_sycl a9a a.m


## CUDA
To build and run the CUDA version of the workload. \
cd cuda \
mkdir build \
cd build \
cmake -DUSE_SM={architecture SM to use for nvidia hardware.. i.e. 80} ../ \
make \

### Running the workloads 
./svm_cuda a9a a.m

## AMD
To build and run the HIP version of the workload. \
cd hip \
mkdir build \
cd build \
cmake ../ \
make \

### Running the workloads 
./svm_hip a9a a.m


