# SeisAcoMod2D (SAM)

Parallel 2D Acoustic Finite Difference Seismic Modelling using the Staggered Grid. The original CUDA source code is from [here](https://github.com/richaras/SeisAcoMod2D). For SYCL version, the CUDA code was migrated using Intel DPCT, and then the resulting code was modified to remove the dpct headers.

## Cloning

To clone, do

```
git clone https://github.com/oneapi-src/Velocity-Bench.git
```
# Build Instructions

## Downlod and extract data:
In addition to the input files in input and geometry folders, there are 4 data files that need to be copied manually (if not already done): 

|Name                           |Size              |
|-------------------------------|------------------|
|sigsbee2a_cp.bin               |15 MB             | 
|sigsbee2a_den.bin              |15 MB             | 
|twolayer_model_cp.bin          |96 MB             | 
|twolayer_model_den.bin         |96 MB             |

The data files can be downloaded from https://github.com/richaras/SeisAcoMod2D/tree/master/data

- Make sure that no folder named 'data' already exists inside the 'SeisAcoMod2D' folder.
- Download the four *.bin files from https://github.com/richaras/SeisAcoMod2D/tree/master/data in a separate location.
- Create a 'data' folder inside 'SeisAcoMod2D' folder (in the same level as 'input' and 'geometry').
- Copy the downloaded *.bin files into the newly created 'data' folder.

```
ls data
```

## To build for SYCL

For Intel GPU -  
First, source icpx compiler. Then,

```
cd SeisAcoMod2D/SYCL
mkdir build
cd build
CXX=mpiicpc cmake -DGPU_AOT=pvc ..
make -sj
```
Note:
- To enable AOT compilation, please use the flag `-DGPU_AOT=pvc` for PVC.

For AMD GPU -  
First source clang++ compiler. Then,
```
cd SeisAcoMod2D/SYCL
mkdir build
cd build
CXX=mpiicpc cmake -DUSE_AMDHIP_BACKEND=gfx90a ..
make -sj
```
Note:
- We use the flag `-DUSE_AMDHIP_BACKEND=gfx90a` for MI250. Use the correct value for your GPU.

For NVIDIA GPU -  
First source clang++ compiler. Then,
```
cd SeisAcoMod2D/SYCL
mkdir build
cd build
CXX=mpiicpc cmake -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=80 ..
make -sj
```
Note:
- We use the flag `-DUSE_SM=80` for A100 or `-DUSE_SM=90` for H100.

## To build for CUDA

```
cd SeisAcoMod2D/CUDA
mkdir build
cd build
CXX=mpiicpc cmake -DUSE_SM=80 ..
make -sj
```

Note:
- We use the flag `-DUSE_SM=80` for A100 or `-DUSE_SM=90` for H100.

## To build for HIP

```
cd SeisAcoMod2D/HIP
mkdir build
cd build
CXX=mpiicpc cmake -DROCM_PATH=/opt/rocm ..
make -sj
```

# Run instructions

After building, to run the workload, cd into the build folder. Then

For running SYCL on PVC 1T
```
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK='0.0' mpiexec -bootstrap ssh -n 2 ./SeisAcoMod2D ../../input/twoLayer_model_5000x5000z_small.json
```
For running SYCL on PVC 2T
```
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK='0' EnableImplicitScaling=1 mpiexec -bootstrap ssh -n 2 ./SeisAcoMod2D ../../input/twoLayer_model_5000x5000z_small.json
```
For running SYCL on NVIDIA GPU
```
ONEAPI_DEVICE_SELECTOR=cuda:gpu mpiexec -bootstrap ssh -n 2 ./SeisAcoMod2D ../../input/twoLayer_model_5000x5000z_small.json
```
For running CUDA on NVIDIA GPU
```
mpiexec -bootstrap ssh -n 2 ./SeisAcoMod2D ../../input/twoLayer_model_5000x5000z_small.json
```
For running SYCL on AMD GPU
```
ONEAPI_DEVICE_SELECTOR=hip:gpu mpiexec -bootstrap ssh -n 2 ./SeisAcoMod2D ../../input/twoLayer_model_5000x5000z_small.json
```
For running HIP on AMD GPU
```
mpiexec -bootstrap ssh -n 2 ./SeisAcoMod2D ../../input/twoLayer_model_5000x5000z_small.json
```

# Output

Output gives the total time (in sec) for running the whole workload.
