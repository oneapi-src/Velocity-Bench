# Tools

## Build & Run 
* Using Intel Compiler **(Heavily recommended)**
    1. Source Intel parallel studio to be able to use ```icpc``` command.
    2. Run configuration script and give it the path to zfp compression directory.
    ```shell script
    ./config.sh -b omp -c <path-to-zfp-compression> --images --tools
    ```
    3. Run the building script.

    ```shell script
    ./clean_build.sh
    ```
  
* Using gcc compiler **(Not Recommended)**
    1. Run configuration script and give it the path to zfp compression directory.
    ```shell script
    ./config.sh -b omp -g -c <path-to-zfp-compression> --images --tools
    ```
    This is will run the RTM using OpenMP technology and gcc compiler and without OpenCV. **For OpenCV, add '--images' to enable OpenCV**.
    For more details about the options for the config script, press [here](https://gitlab.brightskiesinc.com/parallel-programming/reversetimemigration/-/wikis/projects/openmp_rtm/config.sh)
    
    2. Run the building script.
    ```shell script
    ./clean_build.sh
    ```
  
## Available Tools

After configuring and building tools you could then run any of the regarded.
```shell script
./<ToolName> <primary-flags> <optional-flags>
```

### Comparator
```shell script
./Comparator <step> <max-nt> <src> <dst> <csv-prefix>
```
Or
```shell script
./Comparator <src> <dst>
```

### Convertor
Converts from any file format to another
```shell script
./Convertor -d <file-path> -f <type-from> -t <type-to>
```
Or
```shell script
./Convertor -d <file-path> -f <type-from> -t <type-to> -p <percentile>
```

### Generator
Generates synthetic model given meta data in **```*.json```** format file
```shell script
./Generator -m <meta_data.json>
```

#### Generator Meta Data File
Structure goes as the following example:
```json
{
  "model-name": "velocity",
  "is-traces": false,
  "percentile": 98.5,
  "grid-size": {
    "nx": 500,
    "ny": 1,
    "nz": 500,
    "nt": 1
  },
  "cell-dimension": {
    "dx": 6.25,
    "dy": 0,
    "dz": 6.25,
    "dt": 0.005
  },
  "properties": {
    "value-range": {
      "min": 2,
      "max": 32
    },
    "layers": {
      "enable": true,
      "count": 3,
      "type": "sharp"
    },
    "cracks": {
      "enable": true,
      "count": 1
    },
    "salt-bodies": {
      "enable": false,
      "count": 4,
      "type": "random",
      "width": "narrow"
    }
  }
}
```