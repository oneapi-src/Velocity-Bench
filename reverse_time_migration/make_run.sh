#!/bin/bash
#cmake -DCMAKE_BUILD_TYPE=NOMODE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DUSE_OpenMp=OFF -DUSE_DPC=ON -DUSE_OMP_Offload=OFF -DUSE_OpenCV=OFF -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF -DDATA_PATH=data -DWRITE_PATH=results -DUSE_INTEL= -DCOMPRESSION=NO -DCOMPRESSION_PATH=. -DUSE_MPI=OFF '-DCMAKE_CXX_FLAGS=-O3 -std=c++17' -H. -B./bin
#cmake -DCMAKE_BUILD_TYPE=NOMODE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DUSE_OpenMp=OFF -DUSE_DPC=OFF -DUSE_CUDA=ON -DUSE_OMP_Offload=OFF -DUSE_OpenCV=ON -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF -DDATA_PATH=data -DWRITE_PATH=results -DUSE_INTEL= -DCOMPRESSION=NO -DCOMPRESSION_PATH=. -DUSE_MPI=OFF '-DCMAKE_CXX_FLAGS=-O3 -std=c++17' -H. -B./bin

CheckReturnValue() {
    if [ $1 != 0 ]; then
        echo "Aborted"
        exit 1 
    fi
}

if [ "$1" == "" ]; then
    echo "Use ./make_run dpcpp | cuda"
    exit 1
fi 

if [ "$3" == "no_link_check" ]; then
    echo "Skipping data link check"
else
    if [ ! -d "./data" ]; then
        echo "Directory 'data' does not exist!"
        exit 1
    fi
fi

if [ "$2" == "no_build" ]; then
    echo "Skipping build"
    cd ..
else
    cd bin
    make Engine -j
    retval=$?
    CheckReturnValue $retval

    if [ $retval -ne 0 ]; then
        echo "Build failed, abort run"
        exit $retval
    fi

#    make Modeller -j
#    retval=$?
#    CheckReturnValue $retval

    cd -
fi

date=`date '+%Y%m%d_%H_%M_%S'`

if [ -d "./results" ]; then
    mv results results_${date}_INCOMPLETE 

    rm -fr results
    echo "Removing existing results directory"
fi

RESULTS=""

COMPUTE_JSON_FILE="computation_parameters.json"
if [[ "$1" == *"_cpu"* ]]; then
    echo "Using computation_parameters_cpu.json file"
    COMPUTE_JSON_FILE="computation_parameters_cpu.json"
else
    echo "Using computation_parameters.json file"
    COMPUTE_JSON_FILE="computation_parameters.json"
fi

mkdir results
if [ "$1" == "cuda" ] || [ "$1" == "dpcpp_nvidia" ] ; then 
    ./bin/Engine -p workloads/bp_model/${COMPUTE_JSON_FILE}
    RESULTS=`cat results/timing_results.txt | /bin/grep -wi MigrateShot -A4 | /bin/grep -wi Total`
    mv results results_${1}_${date}_complete
elif [ "$1" == "dpcpp" ]; then #PVC
    ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0.0  SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1 ./bin/Engine -p workloads/bp_model/${COMPUTE_JSON_FILE}
    RESULTS=`cat results/timing_results.txt | /bin/grep -wi MigrateShot -A4 | /bin/grep -wi Total`
    mv results results_${1}_${date}_complete
elif [ "$1" == "dpcpp_cpu" ]; then #CPU
    ONEAPI_DEVICE_SELECTOR=opencl:cpu ./bin/Engine -p workloads/bp_model/${COMPUTE_JSON_FILE}
    RESULTS=`cat results/timing_results.txt | /bin/grep -wi MigrateShot -A4 | /bin/grep -wi Total`
    mv results results_${1}_${date}_complete
elif [ "$1" == "dpcpp_amd" ]; then #AMD GPU
    ONEAPI_DEVICE_SELECTOR=HIP ./bin/Engine -p workloads/bp_model/${COMPUTE_JSON_FILE}
    RESULTS=`cat results/timing_results.txt | /bin/grep -wi MigrateShot -A4 | /bin/grep -wi Total`
    mv results results_${1}_${date}_complete
elif [ "$1" == "dpcpp_2t" ]; then #PVC_2T
    if [ -n "$ZE_AFFINITY_MASK" ]; then
        echo "Unsetting affinity mask flag"
        unset ZE_AFFINITY_MASK
    fi

    sed -i -- '0,/normal/s//mpi-static-serverless/g' ./workloads/bp_model/pipeline.json
    export I_MPI_OFFLOAD_DOMAIN_SIZE=1
    export I_MPI_FABRICS=shm:ofi
    export I_MPI_OFFLOAD_TOPOLIB=l0
    export I_MPI_DEBUG=5
    export I_MPI_OFFLOAD_CELL=tile
    export I_MPI_HYDRA_BOOTSTRAP=ssh

    ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1  mpirun -n 2 -ppn 2 ./bin/Engine -p workloads/bp_model/${COMPUTE_JSON_FILE}
    RESULTS=`cat results/timing_results.txt | /bin/grep -wi MigrateShot -A4 | /bin/grep -wi Total`
    mv results results_${1}_mpirun_${date}_complete

    unset I_MPI_OFFLOAD_DOMAIN_SIZE 
    unset I_MPI_FABRICS
    unset I_MPI_OFFLOAD_TOPOLIB
    unset I_MPI_DEBUG
    unset I_MPI_OFFLOAD_CELL
    unset I_MPI_HYDRA_BOOTSTRAP
    echo "Restoring ./workloads/bp_model/pipeline.json"
    git checkout ./workloads/bp_model/pipeline.json
fi

echo "MigrateShot time is : $RESULTS" 
echo ""

