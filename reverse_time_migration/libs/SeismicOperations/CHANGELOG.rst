==============================================
Seismic Operations Release Notes
==============================================

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
* Restructured Seismic Operations library to abid to `Google C++ Style Guide`_.
* Redesigned Seismic Operations library by adding new layer of abstraction.
    * 1. Interface
    * 2. Primitive
    * 3. Concrete
* Interface layer contains the interface only to be implemented.
* Primitive layer contains the common code across different technologies.
* Concrete layer contains the different code for different technologies.
    * Memory transfers
    * Kernels
* Reintroduced configurations' structure.
* Reintroduced configurations' data structures in code.
* Unify constructors of all components to take the same configurations' data structures.

**Fixed**:

* Fixed Modelling Engine.


v2.1.0
=======

**Added**:

* Added ``compile`` stage in CI/CD.
* Added ``prerequisites`` folder.

**Fixed**:

* Fixed CUDA codebase.
* Fixed OneAPI codebase.
* Fixed MPI bug in all variants.
* Fixed OpenMP ``ReversePropagation`` bug.



v2.0.0
=======

**Added**:

* Interfaces and implementation of Seismic Framework:
* Supports migration engines for RTM.
* Supports migration abstract engines (to be implemented) for FWI and PsM
* Supports interfaces for components to be implemented to various techniques/technologies.
    * Boundary Manager
    * Computation Kernel
    * Migration Accommodator
    * Model Handler
    * Source Injector
    * Trace Manager
    * Trace Writer
    * Modelling Configuration Parser
    * Ray Tracer
    * Residual Manager
    * Stoppage Criteria
    * Model Updater
* Data units to be transferred between components are monolithic structs.
* Whole new implementation for a generic GridBox to be used by all algorithms, wave fields approximations and technology
* Supports callbacks.
* Test components' structure implementations.

**Working Features**:

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



v1.0.0
=======

**Added**:

* Interfaces and implementation of basic RTM Framework:
* Supports both modelling and migration engines for RTM.
* Supports interfaces for components to be implemented to various techniques/technologies.
* Data units to be transferred between components are monolithic structs.
* Supports callbacks.
* Helper tools: `dout`, `memory_allocator`.
* Dummy component implementations.


.. _`Google C++ Style Guide`: https://google.github.io/styleguide/cppguide.html#Run-Time_Type_Information__RTTI_).