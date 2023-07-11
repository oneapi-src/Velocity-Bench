# Features


## OpenMP

An optimized OpenMP version

* Support the following boundary conditions:
    * CPML
    * Sponge
    * Random
    * Free Surface Boundary Functionality
* Support the following stencil orders:
    * O(2)
    * O(4)
    * O(8)
    * O(12)
    * O(16)
* Support 2D modeling and imaging
* Support the following algorithmic approaches:
    * Two propagation, an I/O intensive approach where you would store all of the calculated wave fields while performing the forward propagation, then read them while performing the backward propagation.
    * We provide the option to use the ZFP compression technique in the two-propagation workflow to reduce the volume of data in the I/O.
    * Three propagation, a computation intensive approach where you would calculate the forward propagation storing only the last two time steps. You would then do a reverse propagation, propagate the wave field stored from the forward backward in time alongside the backward propagation.
* Support solving the equation system in:
    * Second Order
    * Staggered First Order
    * Vertical Transverse Isotropic (VTI)
    * Tilted Transverse Isotropic (TTI)
* Support manual cache blocking.
  

## OneAPI

An optimized DPC++ version

* Support the following boundary conditions:
    * None
    * Random
    * Sponge
    * CPML
* Support the following stencil orders:
    * O(2)
    * O(4)
    * O(8)
    * O(12)
    * O(16)
* Support 2D modeling  and imaging
* Support the following algorithmic approaches:
    * Three propagation, a computation intensive approach where you would calculate the forward propagation storing only the last two time steps. You would then do a reverse propagation, propagate the wave field stored from the forward backward in time alongside the backward propagation.
* Support solving the equation system in:
    * Second order
    

## CUDA

Basic CUDA version

* Support the following boundary conditions:
    * None
* Support the following stencil orders:
    * O(2)
    * O(4)
    * O(8)
    * O(12)
    * O(16)
* Support 2D modeling  and imaging
* Support the following algorithmic approaches:
    * Three propagation, a computation intensive approach where you would calculate the forward propagation storing only the last two time steps. You would then do a reverse propagation, propagate the wave field stored from the forward backward in time alongside the backward propagation.
* Support solving the equation system in:
    * Second order
