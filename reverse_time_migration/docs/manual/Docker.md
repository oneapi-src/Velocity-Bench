## Docker

This guide assumes:
* You have your terminal opened with the current working directory being the directory of the project.
* Docker engine is installed in your machine.

### OpenMP Docker

1. Change directory to the OpenMP docker file directory
    ```shell script
    cd docker/omp/
    ```
2. Build image from Docker file included in the current directory and give a tag for the image.
    ```shell script
   docker build -t <tag of image> .
    ```
    Note: this step may need sudo privilege.
3. Build a container from the image and give it a name:this step looks for the license file for intel parallel studio in the directory /opt/src/license/ inside the container so replace this directory by the path to the license file in your machine 
by adding ```-v```
    ```shell script
   docker run -v /<path-to>/intel_parrallel_studio_license:/opt/src/license/  --name <name-of-container> -it <tag-of-image> 
    ```


### OneAPI Docker

1. Change directory to the DPC++ docker file directory
    ```shell script
    cd docker/oneapi/
    ```
2. Build image from Docker file included in the current directory and give a tag for the image.
    ```shell script
   docker build -t <tag of image> .
    ```
    Note: this step may need sudo privilege.
3. Build a container from the image and give it a name.
    ```shell script
   docker run --name <name of container> -it <tag of image> 
    ```   


### Additional Options

1.  To enable characterization inside the docker using Intel tools add --privileged=true to the docker run command.
    ```shell script
    docker run --privileged=true --name <name of container> -it <tag of image> 
    ```
2.  To store what is inside a specific directory in docker in an external directory on your machine use the -v option with docker run command.
    ```shell script
    docker run -v  /<path-to>/local directory:/<path-to-directroy-inside-the_docker> --name <name-of-container> -it <tag-of-image>  
    ```
    