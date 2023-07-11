==============================================
Seismic Toolbox Release Notes
==============================================

v3.1.0
=======

**Fixed**:

* Added OpenMP Offload support
* Wave Approximations
    * Isotropic Second Order
* Migration Accommodators
    * Cross Correlation
* Boundary Conditions
    * None (*All*)
* Forward Collectors
    * Reverse Propagation
    * Two Propagation
*  Model Handlers
    * Seismic Model Handler
    * Synthetic Model Handler
* Source Injectors
    * Ricker Source Injector


v3.0.1
=======

**Fixed**:

* CMake now doesn't specify a device for OneAPI
* First touch is now specific to cpu.
* Corrected the computational grid size.
* Fix typo in OneAPI generator.


v3.0.0
=======

**Added**:

* Added CFL condition to OpenMP.
* Added CFL condition to OneAPI.
* Added tests for all components.
* Added Interpolator feature.
* Added Sampler feature.
* Added Compressor feature.
* Added `Thoth` (I/O) library to existing code (N.B. Old I/O still included and should completely be deprecated in later releases)
* Modified some existing features to scale up the code on clusters.
* Renewed `Helpers` library.
* Reintroduced `Generators` module.

**Fixed**:

* Fixed MPI in OpenMP.
* Fixed MPI in OneAPI.
* Fixed bugs in `CorrelationKernel`.
* Fixed bugs in `TraceManager`.
* Fixed 3D support to OpenMP kernels.
* Fixed `CrossCorrelation`'s compensation.
* Fixed `TwoPropagation`.
* Fixed `TwoPropagation` w/Compression.


v2.2.2
=======

**Added**:

* Added 3D support to kernels.
* Added 3D support to I/O.
* Added shots stride for trace reading.


v2.2.1
=======

**Added**:

* Removed ZFP submodule.
* Renewed `Helpers` library
* Changed libraries from `SHARED` to `STATIC`

**Fixed**:

* Fixed testing directives.


v2.2.0
=======

**Added**:

* Restructured CMake.
* Added testing structure.
* Restructured Seismic Toolbox library to abid to `Google C++ Style Guide`_
* Introduced tools directory. Includes various standalone tools to help ease testing.
    * Comparator
    * Convertor
    * Generator
* Split parsers to parsers and generators for better backward compatibility.
* Reintroduced configurations structure.
* Reintroduced configurations data structures in code.
* Added user manual (Initial).


v2.1.0
=======

**Added**: 

* Added ``compile`` stage in CI/CD.
* Added ``prerequisites`` folder.

**Fixed**:

* Fixed CUDA codebase (`#2`_)
* Fixed OneAPI codebase (`#3`_)
* Fixed MPI bug in all variants (`#5`_)
* Fixed OpenMP ``ReversePropagation`` bug (`#7`_)


v2.0.0
=======

**Added**:

* OpenMP working
* Wave Approximations
    * Isotropic First Order
    * Isotropic Second Order
    * VTI First Order
    * TTI First Order
* Migration Accommodators
    * Cross Correlation
    * ADCIG
* Boundary Conditions
    * CPML (*Isotropic First Order / Isotropic Second Order*)
    * Sponge (*All*)
    * None (*All*)
    * Random (*All*)
* Forward Collectors
    * Reverse Propagation
    * Two Propagation
*  Model Handlers
    * Seismic Model Handler
    * Synthetic Model Handler
* Source Injectors
    * Ricker Source Injector

**Bugs**:

*  CUDA and OneAPI broken
*  Modeller needs some final tweaks



.. _`Google C++ Style Guide`: https://google.github.io/styleguide/cppguide.html#Run-Time_Type_Information__RTTI_).
.. _#2: https://gitlab.brightskiesinc.com/parallel-programming/SeismicToolbox/-/issues/2
.. _#3: https://gitlab.brightskiesinc.com/parallel-programming/SeismicToolbox/-/issues/3
.. _#5: https://gitlab.brightskiesinc.com/parallel-programming/SeismicToolbox/-/issues/5
.. _#7: https://gitlab.brightskiesinc.com/parallel-programming/SeismicToolbox/-/issues/7
