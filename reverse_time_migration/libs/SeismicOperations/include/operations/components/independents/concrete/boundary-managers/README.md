# Boundary Managers

All different implementations of the boundary manager interface should reside here. Description of the different
implementations should be below.

## Perfect Refection (No Boundary Condition)

* Component Class : NoBoundaryManager
* Utilizes perfect reflection boundaries(0 velocity in boundaries).
* Supports : 2D & 3D, Full Model and Window Model.

## Random Boundaries

* Component Class : RandomBoundaryManager
* Utilizes random boundaries(defines a random but smooth transition to the velocities in the boundaries).
* Supports : 2D & 3D, Full Model and Window Model.
