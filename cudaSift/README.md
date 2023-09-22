# CudaSift
CudaSift - SIFT features with SYCL, CUDA & HIP

# Building CudaSift
**To build cuda version**

mkdir build && cd build

//For A100 Machine

cmake ../ -DUSE_SM=80

//For H100 Machine

cmake ../ -DUSE_SM=90

make

**To build SYCL version**

mkdir build

cd build

#update the path for OpenCV_DIR

CXX=icpx cmake ../ -DGPU_AOT=pvc

make -sj

**To build SYCL version on NVIDIA Backend**

source /path/to/clang/

mkdir build && cd build

//For A100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=80 

//For H100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=90

make -sj

**To build SYCL version on AMD Backend**

source /path/to/clang/

mkdir build && cd build

//For MI-100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_AMDHIP_BACKEND=gfx908

//For MI-250 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_AMDHIP_BACKEND=gfx90a

make -sj

**To build HIP version**

mkdir build && cd build

CXX=hipcc cmake ../ -DROCM_PATH=/path/to/rocm 
For e.g CXX=hipcc cmake ../ -DROCM_PATH/opt/rocm-5.4.3

make -sj

# Running CudaSift

**To run sycl version**

./cudasift

**To run SYCL on NVIDIA Backend**

./cudaSift

**To run SYCL on AMD Backend**

ONEAPI_DEVICE_SELECTOR=hip:* ./cudaSift

**To run cuda version**

./cudasift

**To run hip version**

./cudasift
