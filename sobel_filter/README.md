# Sobel_Filter

## Description
This workload implements the widely used sobel filter edge detection algorithim in CUDA and SYCL. This is a widely used
algorithm in computer vision and image based machine learning domain, including autonomous driving.

## Downlod and extract data:
sobel_filter has a large image input inside sobel_filter/res folder in .tgz format. Extract the .tgz file and make sure silverfalls_32Kx32K.png is inside the res folder. It can also be copied to some other location in which case the -i flag when running the workload needs to be updated to point to it.  
Note because of the large input image, an ENV variable OPENCV_IO_MAX_IMAGE_PIXELS='1677721600' needs to be in PATH when running the workload. Run instructions below have been udpated for this. Because of the use of large input image, scaling it up (e.g. -f 16) is not needed anymore.
# Build Instructions

## To build for SYCL

For Intel GPU -  
First, source icpx compiler. Then,

```
cd sobel_filter/SYCL
mkdir build
cd build
CXX=icpx cmake -DGPU_AOT=pvc -DOpenCV_DIR=/path/to/opencv/ver/lib/cmake/opencv4 ..
make -sj
```
Note:
- To enable AOT compilation, please use the flag `-DGPU_AOT=pvc` for PVC.

For AMD GPU -  
First source clang++ compiler. Then,
```
cd sobel_filter/SYCL
mkdir build
cd build
CXX=clang++ cmake -DUSE_AMDHIP_BACKEND=gfx90a -DOpenCV_DIR=/path/to/opencv/ver/lib/cmake/opencv4 ..
make -sj
```
Note:
- We use the flag `-DUSE_AMDHIP_BACKEND=gfx90a` for MI250. Use the correct value for your GPU.

For NVIDIA GPU -  
First source clang++ compiler. Then,
```
cd sobel_filter/SYCL
mkdir build
cd build
CXX=clang++ cmake -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=80 -DOpenCV_DIR=/path/to/opencv/ver/lib/cmake/opencv4 ..
make -sj
```
Note:
- We use the flag `-DUSE_SM=80` for A100 or `-DUSE_SM=90` for H100.

## To build for CUDA

```
cd sobel_filter/CUDA
mkdir build
cd build
CXX=nvcc cmake -DUSE_SM=80 -DOpenCV_DIR=/path/to/opencv/ver/lib/cmake/opencv4 ..
make -sj
```

Note:
- We use the flag `-DUSE_SM=80` for A100 or `-DUSE_SM=90` for H100.

## To build for HIP

```
cd sobel_filter/HIP
mkdir build
cd build
CXX=hipcc cmake -DROCM_PATH=/opt/rocm -DOpenCV_DIR=/path/to/opencv/ver/lib/cmake/opencv4 ..
make -sj
```

# Run instructions

After building, to run the workload, cd into the build folder. Then

For running SYCL on PVC 1T
```
OPENCV_IO_MAX_IMAGE_PIXELS='1677721600' ONEAPI_DEVICE_SELECTOR=level_zero:0.0 ./sobel_filter -i ../../res/silverfalls_32Kx32K.png -n 5
```
For running SYCL on PVC 2T
```
OPENCV_IO_MAX_IMAGE_PIXELS='1677721600' ONEAPI_DEVICE_SELECTOR=level_zero:0 EnableImplicitScaling=1 ./sobel_filter -i ../../res/silverfalls_32Kx32K.png -n 5
```
For running SYCL on NVIDIA GPU
```
OPENCV_IO_MAX_IMAGE_PIXELS='1677721600' ONEAPI_DEVICE_SELECTOR=cuda:0 ./sobel_filter -i ../../res/silverfalls_32Kx32K.png -n 5
```
For running SYCL on AMD GPU
```
OPENCV_IO_MAX_IMAGE_PIXELS='1677721600' ONEAPI_DEVICE_SELECTOR=hip:0 ./sobel_filter -i ../../res/silverfalls_32Kx32K.png -n 5
```
For running CUDA on NVIDIA GPU
```
OPENCV_IO_MAX_IMAGE_PIXELS='1677721600' ./sobel_filter -i ../../res/silverfalls_32Kx32K.png -n 5
```
For running HIP on AMD GPU
```
OPENCV_IO_MAX_IMAGE_PIXELS='1677721600' ./sobel_filter -i ../../res/silverfalls_32Kx32K.png -n 5
```

# Output

Output gives the total time (in ms) for running the whole workload.
