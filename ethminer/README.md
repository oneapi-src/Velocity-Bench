# Ethminer

This workload mines ethereum coins using GPUs from Intel, NVIDIA and AMD. For more information, please refer to https://github.com/ethereum-mining/ethminer for further details

### Important notes:

This benchmark requires the following packages to be built on your system (from source) Please use the specified versions
- Boost 1.82.0 (https://www.boost.org/users/history/version_1_82_0.html)
- Json 1.9.5 (https://github.com/open-source-parsers/jsoncpp/releases/tag/1.9.5)
- OpenSSL 1.1.1f (https://github.com/openssl/openssl/releases/tag/OpenSSL_1_1_1f)
- Ethash 0.4.3 (https://github.com/chfast/ethash/releases/tag/v0.4.3)

The commands to build Boost 1.82.0 are the following
```
tar -xvf boost_1.82_0.tar.gz
cd boost_1_82
./bootstrap --prefix=/path/to/boost/install/
./b2 -j`nproc`
./b2 install -j`nproc`
```

The commands to build Json 1.9.5 are the following:
```
tar -xvf jsoncpp_1.9.5.tar.gz
cd jsoncpp_1.9.5
mkdir build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/json/install ..
make -j`nproc`
make install -j
```

The commands to build OpenSSL 1.1.1f are the following:
```
tar -xvf OpenSSL_1_1_1f.tar.gz
cd OpenSSL_1.1.1f
./config --prefix=`realpath /path/to/openssl/install` --openssldir=`realpath /path/to/openssl/install` shared zlib
make -j`nproc`
make install -j
```
The commands to build ethash 0.4.3 are the following:
```
tar -xvf ethash_0.4.3.tar.gz
cd ethash-0.4.3
mkdir build
cmake -DETHASH_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=`realpath /path/to/ethash/install/` ..
make -j`nproc`
make install -j
```

### Build instructions for oneAPI DPC++, (NVIDIA and AMD backend), NVCC/CUDA and ROCM/HIP (Read Carefully)

To compile, you must export the following variables
```
export Boost_DIR=/path/to/boost/1.82.0/
export ethash_DIR=/path/to/ethash/0.4.3
export jsoncpp_DIR=/path/to/json/1.9.5
export OPENSSL_ROOT_DIR=/path/to/openssl/1.1.1f
```

Then, proceed with the following commands for compiling using SYCL (DPC++)

```
mkdir -p ethminer/build
cd build
cmake ..  -DETHASHCUDA=OFF -DETHASHSYCL=ON  
make -j
```
For DPC++ oneAPI compiler, if you want to build for a specific architecture, then add `-DGPU_AOT={pvc|aot}`

## SYCL specific environment variables

- For improved performance, enable immediate command lists by exporting the environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`

## DPC++ using NVIDIA/AMD backend

To compile the code on NVIDIA, please use the following `cmake` command:

```
CC=/path/to/compiler/bin/clang cmake=/path/to/compiler/bin/clang++ .. -DUSE_NVIDIA_BACKEND=ON -DUSE_SM={80|90} -DETHASHSYCL=ON  

```

To compile the code for AMD, please use the following `cmake` command:

```
CC=/path/to/compiler/bin/clang cmake=/path/to/compiler/bin/clang++ .. -DUSE_AMDHIP_BACKEND={gfx90a|gfx908} -DETHASHSYCL=ON  

```

Note: `gfx90a` is for MI250, `gfx908` is for MI100


### Build instructions for CUDA (NVCC)

```
mkdir -p ethminer/build
cd build
cmake .. -DETHASHCUDA=ON -DETHASHSYCL=OFF 
make -j
```

To build for a specific CUDA compute capability, please add `-DUSE_SM=80` or `-DUSE_SM=90` for compute capability 8.0 and 9.0 respectively

### Build instructions for ROCM (HIP)

```
mkdir -p ethminer/build
cd build
CXX=/path/to/rocm/bin/hipcc cmake .. -DETHASHHIP=ON -DETHASHCUDA=OFF -DETHASHSYCL=OFF -DUSE_SYS_OPENCL=OFF -DBINKERN=OFF -DETHASHCL=OFF 
make -j
```

### Run instructions

```
For DPC++ 
ethminer/ethminer -S -M 1 --sy-block-size 1024

For CUDA
ethminer/ethminer -U -M 1

For HIP
ethminer/ethminer --hip -M 1

```
Note: 
- Application will time out in 60 seconds. Please use `--timeout [xx]` to set the number of seconds for application timeout
- The performance metric we use is Mh (e.g., Megahash/s) ` m 11:05:23 ethminer 0:00 A0 3.64 Mh - sy0 3.64` 
- Also, run the workload for 1 minute before breaking/interrupting the program. At the beginning, this will setup the DAG tree, followed by doing a hash 


----------------------------------------------------------------------------------------------------------------------------------------------------

```
$cd ethminer/build
$source /opt/intel/oneapi/setvars.sh
$cmake .. -DETHASHCUDA=OFF -DETHASHSYCL=ON -DUSE_SYS_OPENCL=OFF -DBINKERN=OFF -DETHASHCL=OFF
$make -j

To run the benchmarking:
$ethminer/ethminer -S -M 1
  ethminer 0.19.0+commit.24c5af71.dirty
  Build: linux/relwithdebinfo/gnu
  ...
  Added device Intel(R) Graphics [0x4905]
 i 11:04:38 ethminer Selected pool localhost:0
 i 11:04:38 ethminer Established connection to localhost:0
 i 11:04:38 ethminer Spinning up miners...
sycl 11:04:38 sycl-0   Using Pci Id : Intel(R) Graphics [0x4905]  (Compute ) Memory : 7.45 GB
 i 11:04:38 sim      Epoch : 0 Difficulty : 4.29 Gh
 i 11:04:38 sim      Job: 37ab8d69â€¦ block 1 localhost:0
sycl 11:04:38 sycl-0   Generating DAG + Light(on GPU) : 1.02 GB
  ...
Current device: Intel(R) Graphics [0x4905]
 Max work group size 512
 m 11:04:43 ethminer 0:00 A0 0.00 h - sy0 0.00
 m 11:04:48 ethminer 0:00 A0 0.00 h - sy0 0.00
 m 11:04:53 ethminer 0:00 A0 0.00 h - sy0 0.00
 m 11:04:58 ethminer 0:00 A0 0.00 h - sy0 0.00
sycl 11:05:01 sycl-0   Generated DAG + Light in 23,024 ms. 6.43 GB left.
SYCLMiner::search start_nonce 0
SYCLMiner::search start_nonce 0
 m 11:05:03 ethminer 0:00 A0 87.98 Kh - sy0 87.98
 m 11:05:08 ethminer 0:00 A0 5.00 Mh - sy0 5.00
 m 11:05:13 ethminer 0:00 A0 4.99 Mh - sy0 4.99
 m 11:05:18 ethminer 0:00 A0 4.85 Mh - sy0 4.85
 m 11:05:23 ethminer 0:00 A0 3.64 Mh - sy0 3.64
 m 11:05:28 ethminer 0:00 A0 3.65 Mh - sy0 3.65
 m 11:05:33 ethminer 0:00 A0 3.65 Mh - sy0 3.65
 m 11:05:38 ethminer 0:01 A0 3.64 Mh - sy0 3.64

m 12:25:23 ethminer 1:20 A5 3.63 Mh - sy0 3.63
sycl 12:25:25 sycl-0   Job: 37ab8d69â€¦ Sol: 0x000000041671a635
 i 12:25:25 ethminer **Accepted   1 ms. localhost:0

```

