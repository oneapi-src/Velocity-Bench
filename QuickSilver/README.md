Quicksilver
===========

Introduction
------------

Quicksilver is a proxy application that represents some elements of
the Mercury workload by solving a simpliﬁed dynamic monte carlo
particle transport problem.  Quicksilver attempts to replicate the
memory access patterns, communication patterns, and the branching or
divergence of Mercury for problems using multigroup cross sections.
OpenMP and MPI are used for parallelization.  A GPU version is
available.  Unified memory is assumed.

Performance of Quicksilver is likely to be dominated by latency bound
table look-ups, a highly branchy/divergent code path, and poor
vectorization potential.

For more information, visit the
[LLNL co-design pages.](https://codesign.llnl.gov/quicksilver.php)


Building Quicksilver
--------------------

Instructions to build Quicksilver can be found in the
Makefile. Quicksilver is a relatively easy to build code with no
external dependencies (except MPI and OpenMP).  You should be able to
build Quicksilver on nearly any system by customizing the values of
only four variables in the Makefile:

* CXX The name of the C++ compiler (with path if necessary)
  Quicksilver uses C++11 features, so a C++11 compliant compiler
  should be used.

* CXXFLAGS Command line switches to pass to the C++ compiler when
  compiling objects *and* when linking the executable.

* CPPFLAGS Command line switches to pass to the compiler *only* when
  compiling objects

* LDFLAGS Command line switches to pass to the compiler *only*
  when linking the executable

Sample definitions for a number of common systems are provided.

Quicksilver recognizes a number of pre-processor macros that enable or
disable various code features such as MPI, OpenMP, etc.  These are
described in the Makefile.

For performance measurement run the input file located in Examples/CORAL2_Benchmark/Problem1/Coral2_P1_1.inp

**To build sycl version**

cd SYCL/build

CXX=icpx cmake ../ -DGPU_AOT=PVC

make -sj

**To build sycl version on nvidia backend**

source /path/to/clang/

cd SYCL/build

//For A100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=80

//For H100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=90

make -sj

**To build sycl version on amd backend**

source /path/to/clang/

mkdir build && cd build

//For MI-100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_AMDHIP_BACKEND=gfx908

//For MI-250 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_AMDHIP_BACKEND=gfx90a

make -sj

**To build cuda version**

cd CUDA/src

make USE_SM=90 CUDA_PATH=/opt/hpc_software/compilers/nvidia/cuda-12.0 -sj //for H100

make USE_SM=80 CUDA_PATH=/opt/hpc_software/compilers/nvidia/cuda-12.0 -sj //for A100

**To build hip version**

cd HIP/src

make -sj

Running Quicksilver
-------------------

Quicksilver’s behavior is controlled by a combination of command line
options and an input file.  All of the parameters that can be set on
the command line can also be set in the input file.  The input file
values will override the command line.  Run `$ qs –h` to see
documentation on the available command line switches.  Documentation
of the input file parameters is in preparation.

Quicksilver also has the property that the output of every run is a
valid input file.  Hence you can repeat any run for which you have the
output file by using that output as an input file.

For benchmarking run the example "Examples/CORAL2_Benchmark/Problem1/Coral2_P1_1.inp"

**To run sycl version**

export QS_DEVICE=GPU (This flag is required only for SYCL version)

./qs -i ../../Examples/AllScattering/scatteringOnly.inp

**To run sycl version on nvidia backend**

./qs -i ../../Examples/AllScattering/scatteringOnly.inp

**To run sycl version on amd backend**

ONEAPI_DEVICE_SELECTOR=hip:* ./qs -i ../../Examples/AllScattering/scatteringOnly.inp

**To run cuda version**

./qs -i ../../Examples/AllScattering/scatteringOnly.inp

**To run hip version**

./qs -i ../../Examples/AllScattering/scatteringOnly.inp

License and Distribution Information
------------------------------------

Quicksilver is available [on github](https://github.com/LLNL/Quicksilver)