[![CircleCI](https://circleci.com/gh/LeelaChessZero/lc0.svg?style=shield)](https://circleci.com/gh/LeelaChessZero/lc0)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/3245b83otdee7oj7?svg=true)](https://ci.appveyor.com/project/leelachesszero/lc0)

# Lc0

These instructions are for running and benchmarking the lc0 workload. For the orginal readme file please see the [README_lc0.md](README_lc0.md) file.

## Dependencies
1. The 752187.pb.gz file needs to be placed in the build/release directory. The 752187.pb.gz can be found at the [lc0](https://lczero.org/play/networks/bestnets/) website.
2. The build system for the workload uses the [meson build system](https://mesonbuild.com/). You'll have to download the latest version of meson from [github](https://github.com/mesonbuild/meson) to work with icpx or dpcpp. You'll need to add the directory that conatins meson.py to your environment's path (i.e. export PATH=~/<meson cloned directory>:$PATH ). 

3. Meson also uses [Ninja](https://ninja-build.org/) you can just pip install this to your python environment you're using for the meson.py to build the workload.

## Build Workload 
### SYCL with Nvidia backend.
1. export the LD_LIBRARY_PATH,LIBRARY_PATH, and CPLUS_INCLUDE_PATH for the hipBlas and SYCL compiler. Look at the vars_cuda.sh file for the correct paths. <br/>
2. build lc0 with the matching SM setting for your Nvidia device. <br/>
    `CC=clang CXX=clang++ ./buildSycl.sh -DUSE_NVIDIA_BACKEND=true -DUSE_SM=61 (Set is to whatever architecture your using -i.e. for A100 use 80 for H100 use 90)`
3. Copy the 752187.pb.gz into the build/release directory.    
4. Run the benchmark <br/>
    `cd build/release` <br/>
    `./lc0_sycl benchmark -b sycl`

### SYCL with L0 backend.
1. Source Oneapi.<br/>
2. After sourceing Oneapi build lc0.<br/> 
    `CC=icpx CXX=icpx ./buildSycl.sh -DUSE_L0_BACKEND=true "-DGPU_AOT=['-fsycl-targets=spir64_gen', '-Xs', '-device 0x0bd5 -revision_id 0x2f -options -ze-opt-large-register-file']"`<br/>
3. Run the benchmark.<br/>
    `./lc0_sycl benchmark -b sycl`<br/>
    
### SYCL with AMD backend.
1. export the LD_LIBRARY_PATH,LIBRARY_PATH, and CPLUS_INCLUDE_PATH for the hipBlas and SYCL compiler.<br/>
2. build lc0 with the matching SM setting for your AMD device. <br/>
    `CC=clang CXX=clang++ ./buildSycl.sh -DUSE_AMD_BACKEND=true -DUSE_SM=gfx90a (Set is to whatever architecture your using -i.e. for MI100 use gfx908 for MI250 use gfx90a)`
3. Copy the 752187.pb.gz into the build/release directory.    
4. Run the benchmark <br/>
    `cd build/release` <br/>
    `./lc0_sycl benchmark -b sycl`


### CUDA backend.
1. export the LD_LIBRARY_PATH,LIBRARY_PATH, and CPLUS_INCLUDE_PATH for the cuBlas and SYCL compiler.<br/>
2. build lc0 with the matching SM setting for your Nvidia device. <br/>
    `./buildCuda.sh -DUSE_SM=90`
3. Copy the 752187.pb.gz into the build/release directory.    
4. Run the benchmark <br/>
    `cd build/release` <br/>
    `./lc0_cuda benchmark -b cuda`

### AMD backend.
1. export the LD_LIBRARY_PATH for the cuBlas.<br/>
    `export LD_LIBRARY_PATH=/opt/rocm-5.3.0/lib/rocblas/library:$LD_LIBRARY_PATH`
2. export the LIBRARY_PATH.<br/>
    `export LIBRARY_PATH=/opt/rocm-5.3.0/lib:$LIBRARY_PATH`
3. build lc0 with the matching SM setting for your Nvidia device. <br/>
    `CC=hipcc CXX=hipcc ./buildAMD.sh -DUSE_SM=gfx90a`
4. Copy the 752187.pb.gz into the build/release directory.    
5. Run the benchmark <br/>
    `cd build/release` <br/>
    `./lc0_amd benchmark -b hip`
    
### Output.
#### CUDA Output 
The workload runs twice the first run is just a warmup. 

bestmove e2e4<br/>
Results are correct!<br/>
<br/>
===========================<br/>
<br/>
Total time (ms) : 4869<br/>
Nodes searched  : 33379 <== Report this number. <br/>
Nodes/second    : 6854<br/>
<br/>
<br/>
#### SYCL Output
bestmove e2e4<br/>
Results are correct!<br/>
<br/>
===========================<br/>
Total time (ms) : 7080<br/>
Nodes searched  : 33350 <== Report this number. <br/>
Nodes/second    : 4710<br/>
#### AMD Output
bestmove e2e4<br/>
Results are correct!<br/>
<br/>
===========================<br/>
Total time (ms) : 7080<br/>
Nodes searched  : 33350 <== Report this number. <br/>
Nodes/second    : 4710<br/>

