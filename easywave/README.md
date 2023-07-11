# easyWave

easyWave is a tsunami wave generator developed by ZIB (Original source code from [here](https://github.com/christgau/easywave-sycl)).

## Supported versions

- CUDA: The original code was obtained from [here](https://git.gfz-potsdam.de/id2/geoperil/easyWave)
- DPC++: Currently works on PVC
- HIP: Currently works on AMD Instinct MI100 and MI250 GPUs

Data files used for testing all versions can be found [here](https://git.gfz-potsdam.de/id2/geoperil/easyWave/-/tree/master/data)

# Build Instructions

## DPC++ on PVC

Use the source files from ```easywave/sycl/src```

```
mkdir build
cd build
CC=/path/to/oneAPI/bin/clang CXX=/path/to/oneAPI/bin/clang++ cmake ..
```
Note: To enable AOT compilation, please use the flag `-DGPU_AOT=pvc` for enabling PVC AOT/JIT compilation

## DPC++ using NVIDIA/AMD backend

To compile the SYCL code on NVIDIA GPUs, please use the following:

`-DUSE_NVIDIA_BACKEND=ON -DUSE_SM={80|90}`

To compile the SYCL code on AMD GPUs, please use the following:

`-DUSE_AMDHIP_BACKEND=gfx90a` for MI250 or `-DUSE_AMDHIP_BACKEND=gfx908` for MI100

## CUDA:

Use the source files from ```easywave/CUDA/src```

```
mkdir build
cd build
CC=/path/to/clang/bin/clang CXX=/path/to/clang/bin/clang++ cmake ..
```

For A100 GPU, please use `-DUSE_SM=80` compilation 
For H100 GPU, please use `-DUSE_SM=90` compilation 

Build options through cmake:

* ```-DENABLE_KERNEL_PROFILING``` Enables kernel profiling for CUDA and DPC++

## ROCM:

Use the source files from ```easywave/HIP/src```

```
mkdir build
cd build
CXX=/path/to/rocm/bin/hipcc cmake ..
```

# Run instructions

To run the workload, it is suggested to use the following inputs (as per developer's suggestion)

```
./easywave_{sycl|cuda} -grid /path/to//easywave_data/data/grid/e2Asean.grid -source /path/to/easywave_data/data/faults/BengkuluSept2007.flt -time 120
```
# SYCL specific environment variables

PVC-1T: Please export the following variables `DirectSubmissionOverrideBlitterSupport=2`, `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1`, `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` for increased performance

PVC-2T: Use `EnableImplicitScaling=1`

The profiled time includes memory transfer between device-to-host and vice versa and kernel compute time. File I/O is not included.

# Validation

This benchmark does not have a validation mechanism. It is suggested to use the `eWave.2D.*` output files generated from the CUDA binary using the same input parameters when performing the validation. 

To verify the output, please use the supplied python script from ```easywave/tools``` called ```compare.py``` . 

To use this script, you must use python2.7.9. For example:

```python2.7.9 easywave/tools/compare.py /path/to/cuda/build/eWave.2D.XXXXX.ssh /path/to/dpcpp/build/eWave.2D.XXXXX.ssh```

Each ```eWave.2D.XXXXX.ssh``` file represents the wave at a particular time in seconds. The value of `XXXXX` is the timestamp and it must be the same when comparing the two eWave.2D.XXXXX.ssh files

For example, comparing the values at 07200, execute the following:

```
python2.7.9 easywave/tools/compare.py /path/to/dpcpp/cuda/eWave.2D.07200.ssh /path/to/dpcpp/build/eWave.2D.07200.ssh
Differences: 29399
Max difference: 0.000002
```
