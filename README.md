# SpatialCL - a high performance library for the spatial processing of particles on GPUs

SpatialCL is a library providing an efficient tree data structure for GPUs to spatially index particles in 2D or 3D, as well as optimized parallel query algorithms. Several query algorithms are supported, each optimized for the parallel execution of many independent queries.

## Features
* Fully parallelized for GPUs using OpenCL, including the tree construction
* Both 2D and 3D is supported in both single and double precision
* Apart from their coordinates, particles can carry up to 14 (2D) or 13 (3D) additional floating point values (e.g. masses, charges, velocity, ...)
* The query algorithms are separated from the query handler, which controls what the query actually returns and how it executes. This allows the user to formulate custom queries. During query execution, these custom query handlers are tightly integrated into the existing query algorithms, resulting in extremely low overhead and high performance.
* Out of the box, query handlers for range queries and KNN queries (work in progress) are provided.

Target applications include
* Spatial point/particle processing in general
* Point-cloud based computer graphics
* Scientific applications based on particles, such as N-body simulations or smoothed particle hydrodynamics.

The power of SpatialCL is illustrated by the provided `nbody` example, which implements a complete N-Body simulation using Barnes-Hut gravitational tree code on GPUs using SpatialCL with a custom query handler to efficiently calculate the acceleration of each particle.

## Building SpatialCL

SpatialCL itself is a header-only library, so no special build steps are required to use SpatialCL in your application. In order to compile the examples, benchmarks and tests, execute
```
$ mkdir build
$ cd build
$ cmake <path-to-SpatialCL-directory>
$ make
```
