## Details
`Voxelizer` implements an optimized version of the method described in M. Schwarz and HP Seidel's 2010 paper [*Fast Parallel Surface and Solid Voxelization on GPU's*](http://research.michael-schwarz.com/publ/2010/vox/).

**Dependencies**
The project has the following build dependencies:
 * [Nvidia Cuda 8.0 Toolkit (or higher)](https://developer.nvidia.com/cuda-toolkit) for CUDA + Thrust libraries (standard included)
 * [oneAPI Toolkit] for running SYCL version on Intel GPU
 * [Rocm Library] for running HIP version on AMD GP
 * [Trimesh2](https://github.com/Forceflow/trimesh2) for model importing. Latest version recommended.
 * [GLM](http://glm.g-truc.net/0.9.8/index.html) for vector math. Any recent version will do.
 * [OpenMP](https://www.openmp.org/)

**To build SYCL Version on Intel backend**

mkdir build && cd build

CC=icpx CXX=icpx cmake ../ -DTrimesh2_INCLUDE_DIR="/path/to/trimesh2/include" -DTrimesh2_LINK_DIR="/path/to/trimesh2/lib.Linux64" -DGPU_AOT=PVC

make -sj

**To run SYCL Version on Intel backend** 

ONEAPI_DEVICE_SELECTOR="level_zero:gpu"  ./voxelizer_sycl -f ../../test_models/bunny.OBJ -s 1024 -i 20

**To build SYCL Version on Nvidia backend**

mkdir build && cd build

// For A100
CC=clang CXX=clang++ cmake ../ -DUSE_NVIDIA_BACKEND=YES -DTrimesh2_INCLUDE_DIR="/path/to/trimesh2/include" -DTrimesh2_LINK_DIR="/path/to/trimesh2/lib.Linux64" -DUSE_SM=80

// For H100
CC=clang CXX=clang++ cmake ../ -DUSE_NVIDIA_BACKEND=YES -DTrimesh2_INCLUDE_DIR="/path/to/trimesh2/include" -DTrimesh2_LINK_DIR="/path/to/trimesh2/lib.Linux64" -DUSE_SM=90

**To run SYCL Version on Nvidia backend** 

./voxelizer_sycl -f ../../test_models/bunny.OBJ -s 1024 -i 20

**To build SYCL Version on AMD backend**

mkdir build && cd build

// For MI-100
CC=clang CXX=clang++ cmake ../ -DUSE_AMDHIP_BACKEND=gfx908 -DTrimesh2_INCLUDE_DIR="/path/to/trimesh2/include" -DTrimesh2_LINK_DIR="/path/to/trimesh2/lib.Linux64" -DUSE_SM=80

// For MI-250
CC=clang CXX=clang++ cmake ../ -DUSE_AMDHIP_BACKEND=gfx90a -DTrimesh2_INCLUDE_DIR="/path/to/trimesh2/include" -DTrimesh2_LINK_DIR="/path/to/trimesh2/lib.Linux64" -DUSE_SM=90

**To run SYCL Version on AMD backend** 

ONEAPI_DEVICE_SELECTOR=hip:* ./voxelizer_sycl -f ../../test_models/bunny.OBJ -s 1024 -i 20

**To build CUDA Version**

mkdir build && cd build

cmake -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc -DTrimesh2_INCLUDE_DIR="/path/to/trimesh2/include" -DTrimesh2_LINK_DIR="/path/to/trimesh2/lib.Linux64" -DUSE_SM={compute} ../

make -sj

**To run Cuda version**

./voxelizer_cuda -f ../../test_models/bunny.OBJ -s 1024 -i 20


**To build HIP Version**

The default glm installation which is installed via "apt install libglm-dev" currently doesn't work for HIP version.
Install the library from the source (https://github.com/g-truc/glm/tree/master) and point it that location while compilation.

To build and run HIP version please update these three files in Trimesh library.
Comment "#pragma omp parallel for" in the below three files located under trimesh2/libsrc folder and rebuild the trimesh lib.

libsrc/TriMesh_connectivity.cc
        
libsrc/TriMesh_normals.cc
        
libsrc/TriMesh_pointareas.cc

mkdir build && cd build

CXX=hipcc cmake ../ -DTrimesh2_INCLUDE_DIR=/path/to/trimesh2/include/ -DTrimesh2_LINK_DIR=/path/to/trimesh2/lib.Linux64 -DGLM_INCLUDE_DIR=/path/to/glm

**To run Hip version**

./voxelizer_hip -f ../../test_models/bunny.OBJ -s 1024 -i 20


## Citation
@Voxelizer{cudavoxelizer17,
author = "Jeroen Baert",
title = "Cuda Voxelizer: A GPU-accelerated Mesh Voxelizer",
howpublished = "\url{https://github.com/Forceflow/cuda_voxelizer}",
year = "2017"}
