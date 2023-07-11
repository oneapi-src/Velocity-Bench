# Building & Running

This guide assumes you have your terminal opened with the current working directory being the directory of the project.

## OpenMP Version

Utilizes cache blocking to improve performance, this is provided by the user and might vary according to the model, optimal values were found to be ```5500``` in ```x```, ```55``` in ```z``` on the BP Model.

### Building OpenMP Version
* Using Intel Compiler **(Heavily recommended)**
    1. Source Intel parallel studio to be able to use ```icpc``` command.
    2. Run configuration script and give it the path to zfp compression directory.
    ```shell script
    ./config.sh -b omp -c <path-to-zfp-compression>
    ```
    This is will run the RTM using OpenMP technology and intel compiler and without OpenCV. **For OpenCV, add **```--images```** to enable OpenCV**.
    For more details about the options for the config script, press [here](https://gitlab.brightskiesinc.com/parallel-programming/reversetimemigration/-/wikis/projects/openmp_rtm/config.sh)
    
    3. Run the building script.

    ```shell script
    ./clean_build.sh
    ```
    For more details about the options for the build script, press [here](https://gitlab.brightskiesinc.com/parallel-programming/reversetimemigration/-/wikis/projects/openmp_rtm/clean_build.sh)
* Using gcc compiler **(Not Recommended)**
    1. Run configuration script and give it the path to zfp compression directory.
    ```shell script
    ./config.sh -b omp -g -c <path-to-zfp-compression>
    ```
    This is will run the RTM using OpenMP technology and gcc compiler and without OpenCV. **For OpenCV, add **```--images```** to enable OpenCV**.
    For more details about the options for the config script, press [here](https://gitlab.brightskiesinc.com/parallel-programming/reversetimemigration/-/wikis/projects/openmp_rtm/config.sh)
    
    2. Run the building script.
    ```shell script
    ./clean_build.sh
    ```
    This will build the **```Engine```** (Application for migration) and **```Modeller```** (Application for modelling) in parallel by default.
    For more details about the options for the build script, press [here](https://gitlab.brightskiesinc.com/parallel-programming/reversetimemigration/-/wikis/projects/openmp_rtm/clean_build.sh)


### Run OpenMP
1. Export the compact KMP_AFFINITY for OpenMp threads.
```shell script
export KMP_AFFINITY=compact
```
2. Export the OMP_NUM_THREADS and make it equal to the number of threads wanted to run. For example, here we show if we want to make the rtm run using 36 threads.
```shell script
export OMP_NUM_THREADS=36
```
Or

2. Export the KMP_HW_SUBSET multiplication of cores and threads and make it equal to the number of threads wanted to run. For example, here we show if we want to make the rtm run using 36 threads on 36 cores and 1 thread on each core.
```shell script
export KMP_HW_SUBSET=36c,1t
```
**Warning**:the OMP_NUM_THREADS overrides the KMP_HW_SUBSET values.

3. Run the Processing Engine.
```shell script
./bin/Engine -m <workload-path>
```

---


### Run OpenMP w/MPI
1. In the ```config.sh``` run command you should provide an option **```--mpi```** as follows:
    ```shell script
    ./clean_build.sh -t omp --mpi
    ```
   <b>N.B.</b> **```--mpi```** is just a normal flag same as **```--images```** for image output.
   
2. In your workload you'll find a file named ```pipeline.json```, specify which agent to use. Agents are the fundamental component that takes control of program flow in whatever case it is, be it MPI or Serial approach.
   ```json
   {
     "pipeline": {
       "agent": {
         "type": "mpi-static-server"
       }
     }
   }
   ```
   **N.B.** Available Agents: 
   * ```normal```
   * ```mpi-static-server```
   * ```mpi-static-serverless```
   * ```mpi-dynamic-server```
   * ```mpi-dynamic-serverless```

3. Run the Processing Engine.
```shell script
./bin/Engine -m <workload-path>
```
---


## OneAPI Version

### Building OneAPI Version
1. Source Intel oneAPI to be able to use dpc command. If it is already sourced, this step won't be needed.
2. Run configuration script.
```shell script
./config.sh -b dpc -c <compression-path>
```
This is will compile the **```Engine```** using DPC++ technology and without OpenCV. **For OpenCV, add **```--images```** to enable OpenCV**.
For more details about the options for the config script, press [here](https://gitlab.brightskiesinc.com/parallel-programming/reversetimemigration/-/wikis/projects/openmp_rtm/config.sh)

3. Run the building script.
```shell script
./clean_build.sh
```
This will build the **```Engine```** (Binary for migration) and **```Modeller```** (Binary for modelling) in parallel by default. \

For more details about the options for the build script, press [here](https://gitlab.brightskiesinc.com/parallel-programming/reversetimemigration/-/wikis/projects/openmp_rtm/clean_build.sh)


### Run OneAPI on CPU
```shell script
./bin/Engine -m <workload-path>
```
* Optimal workgroup sizes were found to be ```512``` in ```x```, ```2``` in ```z```.


### Run OneAPI on Gen9 GPU
```shell script
./bin/Engine -p ./workloads/bp_model/computation_parameters_dpc_gen9.json
```
* Optimal workgroup sizes were found to be 128 in x, 16 in z. Notice that for gpu, the x dimension is limited by the maximum workgroup size(256 for Gen9).


---


## CUDA Version

### Building CUDA Version
1. Run configuration script.
```shell script
./config.sh -b cuda -c <compression-path> -g
```
This is will compile the **```Engine```** using CUDA technology and without OpenCV. **For OpenCV, add **```--images```** to enable OpenCV**.

For more details about the options for the config script, press [here](https://gitlab.brightskiesinc.com/parallel-programming/reversetimemigration/-/wikis/projects/openmp_rtm/config.sh)

2. Run the building script.
```shell script
./clean_build.sh
```
This will build the **```Engine```** (Binary for migration) and **```Modeller```** (Binary for modelling) in parallel by default. \

For more details about the options for the build script, press [here](https://gitlab.brightskiesinc.com/parallel-programming/reversetimemigration/-/wikis/projects/openmp_rtm/clean_build.sh)


### Run CUDA
1. Run the Processing Engine.
```shell script
./bin/Engine -m <workload-path>
```
