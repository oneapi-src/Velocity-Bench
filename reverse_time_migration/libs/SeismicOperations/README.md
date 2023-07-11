# Seismic Operations Library

<p>
  <img src="https://img.shields.io/pypi/status/Django.svg" alt="stable"/>
</p>

<p>
This is a modularized structure for the Seismic Operations, this doesn't interface with any other framework and only
provides interfaces and null concrete implementations and documentation of the interfaces and API of the different
seismic components.
</p>

<p>
It also provides the implementation of the engine that will use the components for the full algorithm cycle. This should
be used as a base for any engine oriented variation.
</p>


## Features 

You can find detailed features [here](docs/manual/Features.md).

### Engines

* [RTM](include/operations/engines/concrete/RTMEngine.hpp)
* [FWI](include/operations/engines/concrete/FWIEngine.hpp)
* [PSDM](include/operations/engines/concrete/PSDMEngine.hpp)
* [PSTM](include/operations/engines/concrete/PSTMEngine.hpp)

### Components

* [Boundary Manager](include/operations/components/independents/primitive/BoundaryManager.hpp)
* [Computation Kernel](include/operations/components/independents/primitive/ComputationKernel.hpp)
* [Migration Accommodator](include/operations/components/independents/primitive/MigrationAccommodator.hpp)
* [Model Handler](include/operations/components/independents/primitive/ModelHandler.hpp)
* [Source Injector](include/operations/components/independents/primitive/SourceInjector.hpp)
* [Trace Manager](include/operations/components/independents/primitive/TraceManager.hpp)
* [Trace Writer](include/operations/components/independents/primitive/TraceWriter.hpp)
* [Modelling Configuration Parser](include/operations/components/independents/primitive/ModellingConfigurationParser.hpp)
* [Ray Tracer](include/operations/components/independents/primitive/RayTracer.hpp)
* [Residual Manager](include/operations/components/independents/primitive/ResidualManager.hpp)
* [Stoppage Criteria](include/operations/components/independents/primitive/StoppageCriteria.hpp)
* [Model Updater](include/operations/components/independents/primitive/ModelUpdater.hpp)


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

When installing Seismic Operations, require its version. For us, this is what ```major.minor.patch``` means:

- ```major``` - **MAJOR breaking changes**; includes major new features, major changes in how the whole system works, and complete rewrites; it allows us to _considerably_ improve the product, and add features that were previously impossible.
- ```minor``` - **MINOR breaking changes**; it allows us to add big new features.
- ```patch``` - **NO breaking changes**; includes bug fixes and non-breaking new features.


## Changelog

For previous versions, please see our [CHANGELOG](CHANGELOG.rst) file.

## License
This project is licensed under the The GNU Lesser General Public License, version 3.0 (LGPL-3.0) Legal License - see the [LICENSE](LICENSE.txt) file for details
