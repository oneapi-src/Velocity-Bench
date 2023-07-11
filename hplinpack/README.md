This is a workload for high performance linpack. <br />

## CUDA <br />
Source the oneAPI <br />
cd cuda/hpl-2.3/ <br />
make clean && make <br />
cd bin/intel64/ cp ../../../../datafiles/HPL_small_gpu.dat HPL.dat <br />
export LD_LIBRARY_PATH=../../src/cuda/:$LD_LIBRARY_PATH <br />

## HIP <br />
Source the oneAPI <br />
cd hip/hpl-2.3/ <br />
make clean && make <br />
cd bin/intel64/ cp ../../../../datafiles/HPL_small_gpu.dat HPL.dat <br />
export LD_LIBRARY_PATH=../../src/cuda/:$LD_LIBRARY_PATH <br />

## Open Source oneAPI DPC++ compiler for Nvidia backend <br/>
export USE_AMD_BACKEND=ON <br />
   
Source the oneAPI MPI and Onemkl environment variables. <br />
source /opt/intel/oneapi/mkl/latest/env/vars.sh <br />
source /opt/intel/oneapi/mpi/latest/env/vars.sh <br />

Source the open source oneAPI DPC++ compiler. <br />

cd dpcpp/hpl-2.3/ <br />
make clean && make <br />
cd bin/intel64/ <br />
cp ../../../../datafiles/HPL_small_gpu.dat HPL.dat <br />
export LD_LIBRARY_PATH=../../src/dpcpp/:$LD_LIBRARY_PATH <br />
./xhpl <br />

## Open Source oneAPI DPC++ compiler for Nvidia backend <br/>
export USE_NVIDIA_BACKEND=ON <br />
   
Source the OneAPI MPI and Onemkl environment variables. <br />
source /opt/intel/oneapi/mkl/latest/env/vars.sh <br />
source /opt/intel/oneapi/mpi/latest/env/vars.sh <br />

Source the open source oneAPI DPC++ compiler. <br />
source ~/sycl_workspace/llvm/env.sh <br />

cd dpcpp/hpl-2.3/ <br />
make clean && make <br />
cd bin/intel64/ <br />
cp ../../../../datafiles/HPL_small_gpu.dat HPL.dat <br />
export LD_LIBRARY_PATH=../../src/dpcpp/:$LD_LIBRARY_PATH <br />
./xhpl <br />

## DPC++ MPI version. <br />
source oneAPI <br />
cd dpcpp/hpl-2.3/ <br />
make clean && make <br />
cd bin/intel64/ <br />
cp ../../../../datafiles/HPL_small_gpu_2_tile.dat HPL.dat <br />
export LD_LIBRARY_PATH=../../src/dpcpp/:$LD_LIBRARY_PATH <br />
export I_MPI_DEBUG=5 <br />
export I_MPI_FABRICS=shm <br />
export I_MPI_OFFLOAD_TOPOLIB=level_zero <br />
export I_MPI_OFFLOAD_CELL_LIST=0,1 <br />
mpirun -bootstrap ssh -n 2 ./xhpl <br />

## For CPU. <br />
source oneAPI <br />
export ONEAPI_DEVICE_SELECTOR=opencl:cpu <br />
cd dpcpp/hpl-2.3/ <br />
make clean && make <br />
cd bin/intel64/ <br />
cp ../../../../datafiles/HPL_small_cpu.dat HPL.dat <br />
export LD_LIBRARY_PATH=../../src/dpcpp/:$LD_LIBRARY_PATH <br />
OMP_NUM_THREADS=32, OMP_PLACES=numa_domains, OMP_PROC_BIND=close  ./xhpl <br />

## view output <br />
### look for the GFlops measurement in the output log<br />
================================================================================ <br />
T/V                N    NB     P     Q               Time                 Gflops <br />
-------------------------------------------------------------------------------- <br />
WR10L2L2        4096   768     1     1               0.33              1.387e+02 <br />
-------------------------------------------------------------------------------- <br />
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0056536 ...... PASSED <br />
================================================================================ <br />

Finished      1 tests with the following results: <br />
              1 tests completed and passed residual checks, <br />
              0 tests completed and failed residual checks, <br />
              0 tests skipped because of illegal input values. <br />
-------------------------------------------------------------------------------- <br />


