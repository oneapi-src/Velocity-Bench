# Advanced Running Options

## Program Arguments
```
Usage : ./bin/Engine

Performs reverse time migration according to the parameters parsed from the different configuration files.

Optional flags : 
	-m <workload-path>
	Workloads configurations path.
	Default is "./workloads/bp_model"
	
	-p <computation-parameter-file>
	Computation parameter configurations path.
	Default is "./workloads/bp_model/computation_parameters.json"
	
	-s <engine-configuration-file> : 
	Engine configurations configurations path.
	Default is "./workloads/bp_model/engine_configuration.json"
	
	-c <callback-configuration-file> : 
	Callbacks configurations file path. 
	Default is "./workloads/bp_model/callback_configuration.json"
	
	-w <write-path> :
	Results write path.
	Default is "./results"
	
	-h
	Print the options for this command,
```


## Configuration Files

All configuration files are ```*.json``` files which has a generic structure for a user friendly experience.

### Structure
```json
{
  "<BLOCK>": {
    "<ATTRIBUTE>": "<VALUE>"
    }
}
```
You can either add all blocks to one file or separate them to the following structured files.


### Computation Parameter Configuration Block
The computation parameter configuration file is a JSON file which has the following structure:
```json
{
  "computation-parameters": {
    "stencil-order": "8",
    "boundary-length": "20",
    "source-frequency": "20",
    "dt-relax": "0.9",
    "algorithm": "cpu",
    "device": "none",
    "cache-blocking": {
      "block-x": "5500",
      "block-z": "55",
      "block-y": "1",
      "cor-block": "256"
    },
    "window": {
      "enable": "yes",
      "left": "600",
      "right": "1300",
      "depth": "0",
      "front": "0",
      "back": "0"
    }
  }
}
```

<br>

#### Computation Parameters Block

**```stencil-order```**\
Signifies the order of the approximation of the finite difference for space derivatives. The supported values are ```2```, ```4```, ```8```, ```12``` and  ```16```.

**```boundary-length```**\
Is a number that signifies the boundary layer thickness, can have any integer value ```>= 0```.

**```source-frequency```**\
Should be specified in Hz. Should be a value ```> 0```.

**```dt-relax```**\
Is the factor to be multiplied in the dt calculated by the stability criteria as an extra measure of safety, should be > 0 and < 1, normally 0.9.

**```block-x```, ```block-z``` and ```block-y```**\
These parameters control the cache blocking in OpenMP and the workgroup/elements per workitem in DPC++, they have different constraints according to the device or technology used (The constraint is told in the running part for each device).

**```cor-block```**\
Is a DPC++ only parameter that controls the workgroup size for the correlation operation.

**```algorithm```**\
Is a DPC++ only parameter that can take the value of ```cpu```, ```gpu```, ```gpu-semi-shared``` and ```gpu-shared```. 
The different gpu options will select different kernel optimizations to run. Both ```gpu``` and ```gpu-shared``` give the best performance when the blocking is tuned correctly.

**```device```**\
Is a DPC++ only parameter that can either be none, making device selection be using default selector according to type of the algorithm chosen. A pattern can also be provided to make a custom device selection and choose a certain device. Example: ```"device" : "Gen9"``` to specifically select Intel Gen9 Graphics card.

<br>

#### Window Block
**```enable```**\
Is an option that enables windowing migration, this make the migration of each shot only happen in a sub-domain taken from the full domain according to window parameters, around the source of each shot. This contributes to large speedups, but might affect accuracy if not careful in selecting the window. Supported options are <```yes```> and <```no```>.

**```left-window```**\
The window to take left of the source point in x-axis.
 
**```right-window```**\
The window to take right of the source point in x-axis.

**```depth-window```**\
The window to take below of the source point in z-axis.

**```back-window```**\
The window to take behind of the source point in y-axis(Only effective in 3D, not yet supported).

**```front-window```**\
The window to take in front of the source point in y-axis(Only effective in 3D, not yet supported).

\
**N.B.** A sample of this file is available in 'workloads/bp_model/computation_parameters.txt'.


### Engines Configurations Block
The main engine configuration file will define the properties to use in the different computations. A sample of this file is available in **```'workloads/bp_model/engine_configuration.json'```**.

```json
{
  "traces": {
    "min": "601",
    "max": "602",
    "sort-type": "CSR",
    "paths": [
        "data/shots0001_0200.segy",
        "data/shots0201_0400.segy",
        "data/shots0401_0600.segy",
        "data/shots0601_0800.segy",
        "data/shots0801_1000.segy",
        "data/shots1001_1200.segy",
        "data/shots1201_1348.segy"
    ]
  },
  "models": {
    "velocity": "data/velocity.segy",
    "density": "data/density.segy"
  },
  "wave": {
    "physics": "acoustic",
    "approximation": "isotropic",
    "equation-order": "second",
    "grid-sampling": "uniform"
  },
  "components": {
    "boundary-manager": {
      "type": "random",
      "use-top-layer": "yes"
    },
    "migration-accommodator": {
      "type": "cross-correlation",
      "compensation": "combined"
    },
    "forward-collector": {
      "type": "three"
    },
    "trace-manager": {
      "type": "segy",
      "interpolation": "none"
    },
    "source-injector": {
      "type": "ricker"
    },
    "model-handler": {
      "type": "segy"
    }
  }
}
```

#### Models Block
Models file as indicated in the engine configuration. It goes with the following pattern:
```json
{
  "traces": {
    "min": "<value>",
    "max": "<value>",
    "sort-type": "<type>",
    "paths": [
        "<file-path>",
        "<file-path>",
        "<file-path>"
    ]
  }
}
```

#### Traces Block
Traces file as indicated in the engine configuration. It goes with the following pattern:
```json
{
  "models": {
    "<parameter-name>" : "<file-path>"
  }
}
```

* This is a block comprised of multiple attributes:
    * Minimum shot id to start migration from. This is inclusive meaning the shot with the corresponding number will be processed. If we have no minimum limit wanted, write none on the first line.
    * Maximum shot id to stop processing after. This is an inclusive meaning the shot with the corresponding number will be processed. If we have no maximum limit wanted, write none on the second line.
    * Multiple lines each having a trace file path.
* The previous example is for a valid traces configuration file which will process shot 601 and 602


### Callback Configuration Block
* Callback configuration file to produce intermediate files for visualization or value tracking. A sample of this file is available in **```'workloads/bp_model/callback_configuration.json'```**.
* Note: images will be generated only if opencv is enabled in the configurations(./config.sh -i on)
```json
{
  "callbacks": {
    "su": {
      "enable": "yes",
      "show_each": "200",
      "write-in-little-endian": "no"
    },
    "csv": {
      "enable": "yes",
      "show_each": "200"
    },
    "image": {
      "enable": "yes",
      "show-each": "200",
      "percentile": "98.5"
    },
    "norm": {
      "enable": "no",
      "show-each": "100"
    },
    "binary": {
      "enable": "no",
      "show-each": "200"
    },
    "segy": {
      "enable": "yes",
      "show-each": "200"
    },
    "writers": {
      "migration": {
        "enable": "yes"
      },
      "velocity": {
        "enable": "yes"
      },
      "traces-raw": {
        "enable": "yes"
      },
      "traces-preprocessed": {
        "enable": "no"
      },
      "re-extended-velocity": {
        "enable": "yes"
      },
      "each-stacked-shot": {
        "enable": "no"
      },
      "single-shot-correlation": {
        "enable": "no"
      },
      "backward": {
        "enable": "yes"
      },
      "forward": {
        "enable": "yes"
      },
      "reverse": {
        "enable": "yes"
      }
    }
  }
}
```

* For visualization of the **```.segy```** / **```.su```** files, seismic unix can be used.
    ```shell script
    segyread tape=file.segy | suximage perc=98.5
    ```
    ```shell script
    suximage < file.su perc=98.5
    ````
    
* For visualization of the **```.bin```** files, ximage can be used directly.
    ```shell script
    ximage < file.bin n1=195
    ```
    notice in binary you'd need to provide the trace length in the command to visualize it
