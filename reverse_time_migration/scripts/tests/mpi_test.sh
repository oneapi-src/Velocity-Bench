#!/bin/bash

#making the results directory
mkdir No_MPI
mkdir Static_no_server
mkdir Dynamic_no_server
mkdir Static
mkdir Dynamic

source /opt/intel/parallel_studio_xe_2020/psxevars.sh

#Configuration and Building of the Program
./config.sh -k ON -t omp -r -c ~/zfp-0.5.5
./clean_build.sh
export KMP_AFFINITY=compact

#Running the No_MPI version
./bin/acoustic_engine
cp -r ./results ./No_MPI

#Running the MPI static with no server approach using only one MPI process
mpirun -n 1 -genv I_MPI_PIN_DOMAIN=omp ./bin/acoustic_engine_mpi_no_master
cp -r ./results ./Static_no_server

#Comparing MPI static with no server approach with No_MPI
./compare_csv ./No_MPI/results/raw_migration.csv ./Static_no_server/results/raw_migration.csv >Static_no_master.txt
Static_no_master_comparison=$(grep "File comparision" Static_no_master.txt)
Static_no_master_comparison="${Static_no_master_comparison/File comparision/ }"
echo $Static_no_master_comparison | cut -d' ' -f 1 >L2_Error.txt
echo $Static_no_master_comparison | cut -d' ' -f 2 >Absolute_Error.txt

#Running the MPI dynamic with all working approach using only one MPI process
mpirun -n 1 -genv I_MPI_PIN_DOMAIN=omp ./bin/acoustic_engine_mpi_dynamic_no_master
cp -r ./results ./Dynamic_no_server

#Comparing MPI dynamic with all working approach with No_MPI
./compare_csv ./No_MPI/results/raw_migration.csv ./Dynamic_no_server/results/raw_migration.csv >Dynamic_no_server.txt
Dynamic_no_master_comparison=$(grep "File comparision" Dynamic_no_server.txt)
Dynamic_no_master_comparison="${Dynamic_no_master_comparison/File comparision/ }"
echo $Dynamic_no_master_comparison | cut -d' ' -f 1 >>L2_Error.txt
echo $Dynamic_no_master_comparison | cut -d' ' -f 2 >>Absolute_Error.txt

#Running the MPI static with server approach using only one MPI process
mpirun -n 2 -genv I_MPI_PIN_DOMAIN=omp ./bin/acoustic_engine_mpi
cp -r ./results ./Static

#Comparing MPI static with server approach with No_MPI
./compare_csv ./No_MPI/results/raw_migration.csv ./Static/results/raw_migration.csv >Static.txt
Static_comparison=$(grep "File comparision" Static.txt)
Static_comparison="${Static_comparison/File comparision/ }"
echo $Static_comparison | cut -d' ' -f 1 >>L2_Error.txt
echo $Static_comparison | cut -d' ' -f 2 >>Absolute_Error.txt

#Running the MPI dynamic with server approach using only one MPI process
mpirun -n 2 -genv I_MPI_PIN_DOMAIN=omp ./bin/acoustic_engine_mpi_dynamic
cp -r ./results ./Dynamic

#Comparing MPI dynamic with server approach with No_MPI
./compare_csv ./No_MPI/results/raw_migration.csv ./Dynamic/results/raw_migration.csv >Dynamic.txt
Dynamic_comparison=$(grep "File comparision" Dynamic.txt)
Dynamic_comparison="${Dynamic_comparison/File comparision/ }"
echo $Dynamic_comparison | cut -d' ' -f 1 >>L2_Error.txt
echo $Dynamic_comparison | cut -d' ' -f 2 >>Absolute_Error.txt
