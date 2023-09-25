# Reverse Time Migration 

### Notes

This workload depends on the following:
- OpenCV 4.5.5 for outputting intermediate images
- MPI for 2 tile scaling

This workload was developed by Brightskies in DPC++. The oneAPI performance team ported the code into CUDA because there is no CUDA version available publicly. To build, you have two options: the `./config.sh` script, or use the `cmake` commands directly

# Build instructions using `cmake`:

To configure with `cmake` (Please read carefully)

1. To do a clean build, you must remove the `bin` directory
2. You must create a `results` directory that's empty

## Build for oneAPI DPC++ compiler with Intel GPU: 

``` 
CC=/path/to/icpx CXX=/path/to/icpx cmake -DCMAKE_BUILD_TYPE=NOMODE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_DPC=ON -DUSE_NVIDIA_BACKEND=OFF -DGPU_AOT= -DUSE_CUDA=OFF -DUSE_SM= -DUSE_OpenCV=ON -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF -DDATA_PATH=data -DWRITE_PATH=results -DUSE_INTEL= -DCOMPRESSION=NO -DCOMPRESSION_PATH=. -DUSE_MPI=ON -H. -B./bin
cd bin
make Engine -j ### Engine binary is only needed
```

Note: Be sure that `-DUSE_MPI=ON`, `-DUSE_OpenCV=ON`, `-DUSE_DPC=ON` are set. Other flags should be set to `OFF`


## To build on NVIDIA-BACKEND:

```
CC=/path/to/intel/llvm/clang CXX=/path/to/intel/llvm/clang++ cmake -DCMAKE_BUILD_TYPE=NOMODE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_DPC=ON -DUSE_NVIDIA_BACKEND=YES -DGPU_AOT= -DUSE_CUDA=OFF -DUSE_SM={80|90} -DUSE_OpenCV=ON -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF -DDATA_PATH=data -DWRITE_PATH=results -DUSE_INTEL= -DCOMPRESSION=NO -DCOMPRESSION_PATH=. -DUSE_MPI=OFF -H. -B./bin
cd bin
make -j ### Engine binary is only needed
```
Note: Be sure that `-DUSE_DPC=ON`, `-DUSE_NVIDIA_BACKEND=YES`, `-DUSE_OpenCV=ON` are set. Other flags should be `OFF`
To compile for 8.0 or 9.0 compute capability, please use `-DUSE_SM=80` or `-DUSE_SM=90` respectively

## To build on AMD-BACKEND:

```
CC=/path/to/intel/llvm/clang CXX=/path/to/intel/llvm/clang++ cmake -DCMAKE_BUILD_TYPE=NOMODE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_DPC=ON -DUSE_NVIDIA_BACKEND=OFF -DUSE_AMD_BACKEND=SPECIFY_AMD_GPU_ARCHITECTURE_HERE -DGPU_AOT= -DUSE_CUDA=OFF -DUSE_SM= -DUSE_OpenCV=ON -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF -DDATA_PATH=data -DWRITE_PATH=results -DUSE_INTEL= -DCOMPRESSION=NO -DCOMPRESSION_PATH=. -DUSE_MPI=OFF -H. -B./bin
cd bin
make Engine -j ### Engine binary is only needed
```
Note: Be sure that `-DUSE_DPC=ON`, `-DUSE_AMD_BACKEND=[SPECIFY AMD GPU ARCHITECTURE HERE]`, `-DUSE_OpenCV=ON` are set. The AMD gpu architecture we tested are `gfx900` (Vega-FE) and `gfx908` (MI100)

## Build for NVCC Compiler:

```
cmake -DCMAKE_BUILD_TYPE=NOMODE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_DPC=OFF -DUSE_NVIDIA_BACKEND=OFF -DGPU_AOT= -DUSE_CUDA=ON -DUSE_SM=80 -DUSE_OpenCV=ON -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF -DDATA_PATH=data -DWRITE_PATH=results -DUSE_INTEL= -DCOMPRESSION=NO -DCOMPRESSION_PATH=. -DUSE_MPI=OFF -H. -B./bin
cd bin

make Engine -j ### Engine binary is only needed
```
Note: Be sure that `-DUSE_CUDA=ON`, `-DUSE_SM=80 [Or another compute capability]`, `-DUSE_OpenCV=ON` are set. Other flags should be set to `off`

## Build for ROCM/HIP Compiler:

```
CXX=/path/to/rocm/bin/hipcc -DCMAKE_BUILD_TYPE=NOMODE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_DPC=OFF -DUSE_NVIDIA_BACKEND=OFF -DGPU_AOT= -DUSE_CUDA=OFF -DUSE_HIP=ON -DUSE_SM= -DUSE_OpenCV=ON -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF -DDATA_PATH=data -DWRITE_PATH=results -DUSE_INTEL= -DCOMPRESSION=NO -DCOMPRESSION_PATH=. -DUSE_MPI=OFF -H. -B./bin
cd bin

make Engine -j ### Engine binary is only needed
```

##

### Get and setup data files

1. Go to `prerequisites/data-download` directory
2. Run the `./download_bp_data_iso.sh` script. This will download all the necessary `.segy` files 

# Running the workload using command lines directly

Before you run the workload, make sure the `results` directory is created and is empty

To run the workload: `./bin/Engine -p workloads/bp_model/computation_parameters.json`

## Note for PVC only

Please `export SYCL_PI_LEVEL_ZERO_BATCH_SIZE=1000, SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1, SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1` for better performance

To run the workload using MPI (PVC) Only:
1. Modify the ./workloads/bp_model/pipeline.json
2. Set the `type` fields under `pipeline` and `writer` to : mpi-static-serverless
3. Export the following `MPI_` variables:

```
export I_MPI_OFFLOAD_DOMAIN_SIZE=1
export I_MPI_FABRICS=shm:ofi
export I_MPI_OFFLOAD_TOPOLIB=l0
export I_MPI_DEBUG=5
export I_MPI_OFFLOAD_CELL=tile
export I_MPI_HYDRA_BOOTSTRAP=ssh
```

4. Run the workload (for PVC): `mpirun -n 2 -ppn 2 ./bin/Engine -p workloads/bp_model/computation_parameters_pvc.json`

# Running the workload by using the `./make_run.sh` script (simplified)

This script encapsulates all the environment variables needed to run the workload so that the steps listed above are automated.

### Run using 1-Tile (Uses computation_parameters.json)

Execute: `./make_run.sh dpcpp` , This will set the `ZE_AFFINITY_MASK=0.0` automatically, then `grep` for `MigrateShot` which is the total time needed to execute the migration

### Run using 2-Tile (Uses computation_parameters.json)

Execute: `./make_run.sh dpcpp_2t`, This will UNSET ZE_AFFINITY_MASK if it was set, then sets the necessary `I_MPI_*` variables (please see the script). 
Also note, that the script automatically modifies the `workloads/bp_model/pipeline.json` file for executing the workload in `mpi-static-serverless` mode 2T scaling

### Run CUDA A100

Execute: `./make_run.sh cuda`

###

# Seismic Toolbox

<p>
Seismic Toolbox contains all different seismology algorithm (RTM currently). Algorithms are computationally intensive processes which requires propagating wave in 2D model using time domain finite differences wave equation solvers.
</p>

<p>
During the imaging process a forward-propagated source wave field is combined at regular time steps with a back-propagated receiver wave field. Traditionally, synchronization of both wave fields result in a very large volume of I/O, disrupting the efficiency of typical supercomputers. Moreover, the wave equation solvers are memory bandwidth bound due to low flop-per-byte ratio and non-contiguous memory access, resulting hence in a low utilization of available computing resources.
</p>

<p>
Alternatively, approaches to reduce the IO bottleneck or remove it completely to fully utilize the processing power are usually explored and utilized such as the use of compression to reduce the I/O volume. Another approach that eliminates the need for I/O would be to add another propagation in reverse-time to the forward propagated source wave field.
</p>

## Table of Contents
- [Features](#Features)
- [Prerequisites](#Prerequisites)
- [Setup The Environment](#Setup The Environment)
- [Docker](docs/manual/Docker.md#Docker)
    - [OpenMP docker](docs/manual/Docker.md#OpenMP Docker)
    - [OneAPI docker](docs/manual/Docker.md#OneAPI Docker)
    - [Additional Options](docs/manual/Docker.md#Additional Options)
- [Building & Running](docs/manual/BuildingAndRunning.md#Building-&-Running)
    - [OpenMP Version](docs/manual/BuildingAndRunning.md#OpenMP Version)
        - [Building OpenMP Version](docs/manual/BuildingAndRunning.md#Building OpenMP Version)
        - [Run OpenMP](docs/manual/BuildingAndRunning.md#Run OpenMP)
    - [OneAPI Version](docs/manual/BuildingAndRunning.md#OneAPI-version)
        - [Building OneAPI Version](docs/manual/BuildingAndRunning.md#building-OneAPI-version)
        - [Run OneAPI on CPU](docs/manual/BuildingAndRunning.md#Run OneAPI on CPU)
        - [Run OneAPI on Gen9 GPU](docs/manual/BuildingAndRunning.md#Run OneAPI on Gen9 GPU)
    - [CUDA Version](docs/manual/BuildingAndRunning.md#CUDA Version)
        - [Building CUDA Version](docs/manual/BuildingAndRunning.md#Building CUDA Version)
        - [Run CUDA](docs/manual/BuildingAndRunning.md#Run CUDA)
- [Advanced Running Options](docs/manual/AdvancedRunningOptions.md#Advanced-Running-Options)
    - [Program Arguments](docs/manual/AdvancedRunningOptions.md#Program Arguments)
    - [Configuration Files](docs/manual/AdvancedRunningOptions.md#Configuration Files)
        - [Structure](docs/manual/AdvancedRunningOptions.md#Structure)
        - [Computation Parameter Configuration Block](docs/manual/AdvancedRunningOptions.md#Computation Parameter Configuration Block)
        - [Engines Configurations Block](docs/manual/AdvancedRunningOptions.md#Engines Configurations Block)
        - [Callback Configuration Block](docs/manual/AdvancedRunningOptions.md#Callback Configuration Block)
- [Results Directories](docs/manual/ResultsDirectories.md#Results-Directories)
- [Tools](docs/manual/Tools.md#Tools)
    - [Build & Run](docs/manual/Tools.md#Build-&-Run)
    - [Available Tools](docs/manual/Tools.md#Available-Tools)
        - [Comparator](docs/manual/Tools.md#Comparator)
        - [Generator](docs/manual/Tools.md#Generator)
- [Versioning](#Versioning)
- [Changelog](#Changelog)
- [License](#License)


## Features

* An optimized OpenMP version:
    * Support the following boundary conditions:
        * CPML
        * Sponge
        * Random
        * Free Surface Boundary Functionality
    * Support the following stencil orders:
        * O(2)
        * O(4)
        * O(8)
        * O(12)
        * O(16)
    * Support 2D modeling and imaging
    * Support the following algorithmic approaches:
        * Two propagation, an I/O intensive approach where you would store all of the calculated wave fields while performing the forward propagation, then read them while performing the backward propagation.
        * We provide the option to use the ZFP compression technique in the two-propagation workflow to reduce the volume of data in the I/O.
        * Three propagation, a computation intensive approach where you would calculate the forward propagation storing only the last two time steps. You would then do a reverse propagation, propagate the wave field stored from the forward backward in time alongside the backward propagation.
    * Support solving the equation system in:
        * Second Order
        * Staggered First Order
        * Vertical Transverse Isotropic (VTI)
        * Tilted Transverse Isotropic (TTI)
    * Support manual cache blocking.
* An optimized DPC++ version:
    * Support the following boundary conditions:
        * None
        * Random
        * Sponge
        * CPML
    * Support the following stencil orders:
        * O(2)
        * O(4)
        * O(8)
        * O(12)
        * O(16)
    * Support 2D modeling  and imaging
    * Support the following algorithmic approaches:
        * Three propagation, a computation intensive approach where you would calculate the forward propagation storing only the last two time steps. You would then do a reverse propagation, propagate the wave field stored from the forward backward in time alongside the backward propagation.
    * Support solving the equation system in:
        * Second order
* Basic CUDA version:
    * Support the following boundary conditions:
        * None
    * Support the following stencil orders:
        * O(2)
        * O(4)
        * O(8)
        * O(12)
        * O(16)
    * Support 2D modeling  and imaging
    * Support the following algorithmic approaches:
        * Three propagation, a computation intensive approach where you would calculate the forward propagation storing only the last two time steps. You would then do a reverse propagation, propagate the wave field stored from the forward backward in time alongside the backward propagation.
    * Support solving the equation system in:
        * Second order


## Setup The Environment
1. Clone the basic project
    ```shell script
    git clone https://gitlab.brightskiesinc.com/parallel-programming/SeismicToolbox
    ```

2. Change directory to the project base directory
    ```shell script
    cd SeismicToolbox/
    ```
3. To install and download everything you can easily run the ```setup.sh``` script found in ```/prerequisites``` folder
    ```shell script
    ./prerequisites/setup.sh
    ```
   or refer to the ```README.md``` file in ```/prerequisites``` folder for more specific installations. 


## Prerequisites
* **CMake**\
```CMake``` version 3.5 or higher.

* **C++**\
```c++11``` standard supported compiler.

* **Catch2**\
Already included in the repository in ```prerequisites/catch```

* **OneAPI**\
[OneAPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html) for the DPC++ version.

* **ZFP Compression**
    * Only needed with OpenMp technology
    * You can download it from a script found in ```prerequisites/utils/zfp``` folder
      
* **OpenCV**
    * Optional
    * v4.3 recommended
    * You can download it from a script found in ```prerequisites/frameworks/opencv``` folder


## Versioning

When installing Seismic Toolbox, require its version. For us, this is what ```major.minor.patch``` means:

- ```major``` - **MAJOR breaking changes**; includes major new features, major changes in how the whole system works, and complete rewrites; it allows us to _considerably_ improve the product, and add features that were previously impossible.
- ```minor``` - **MINOR breaking changes**; it allows us to add big new features.
- ```patch``` - **NO breaking changes**; includes bug fixes and non-breaking new features.


## Changelog

For previous versions, please see our [CHANGELOG](CHANGELOG.rst) file.


## License
This project is licensed under the The GNU Lesser General Public License, version 3.0 (LGPL-3.0) Legal License - see the [LICENSE](LICENSE.txt) file for details

