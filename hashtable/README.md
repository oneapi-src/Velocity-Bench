# hashtable

hashtable implements a simple hash table in GPU (original CUDA source code is from [here](https://github.com/nosferalatu/SimpleGPUHashTable)).

## Supported versions

- CUDA: The original code was obtained from [here](https://github.com/nosferalatu/SimpleGPUHashTable)
- SYCL: The CUDA code was migrated using Intel DPCT, and then the resulting code was modified to remove the dpct headers.
- HIP: Created from CUDA version using hipify-perl script.


# Build Instructions

## To build for SYCL

For Intel GPU -  
First, source icpx compiler. Then,

```
cd hashtable/SYCL
mkdir build
cd build
CXX=icpx cmake -DGPU_AOT=pvc ..
make -sj
```
Note:
- To enable AOT compilation, please use the flag `-DGPU_AOT=pvc` for PVC.

For AMD GPU -  
First source clang++ compiler. Then,
```
cd hashtable/SYCL
mkdir build
cd build
CXX=clang++ cmake -DUSE_AMDHIP_BACKEND=gfx90a ..
make -sj
```
Note:
- We use the flag `-DUSE_AMDHIP_BACKEND=gfx90a` for MI250. Use the correct value for your GPU.

For NVIDIA GPU -  
First source clang++ compiler. Then,
```
cd hashtable/SYCL
mkdir build
cd build
CXX=clang++ cmake -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=80 ..
make -sj
```
Note:
- We use the flag `-DUSE_SM=80` for A100 or `-DUSE_SM=90` for H100.

## To build for CUDA

```
cd hashtable/CUDA
mkdir build
cd build
CXX=nvcc cmake -DUSE_SM=80 ..
make -sj
```

Note:
- We use the flag `-DUSE_SM=80` for A100 or `-DUSE_SM=90` for H100.

## To build for HIP

```
cd hashtable/HIP
mkdir build
cd build
CXX=hipcc cmake -DROCM_PATH=/opt/rocm ..
make -sj
```

# Run instructions

After building, to run the workload, cd into the build folder. Then

For SYCL:
```
./hashtable_sycl
```
For CUDA:
```
./hashtable_cuda
```

For HIP:
```
./hashtable_hip
```

## Coomand line options
- `--seed seeding_number`: To recreate the same random numbers across runs of the program.
- `--no-verify`: To disable verification for the results.

# Output

Output gives number of keys per second.
