# cuda_voxelizer v0.1
Experimental CUDA voxelizer, to convert polygon meshes to annotated voxel grids. 
 * Outputs data to [.binvox file format](http://www.patrickmin.com/binvox/binvox.html) (default) or a morton-ordered grid. More output formats (magicavoxel, minecraft schematic) are in development.
 * Requires a CUDA-compatible video card. Compute Capability 2.0 or higher (Nvidia Fermi or better).
 
 **Note:** The latest CUDA version (9.1) does not support the compiler changes introduced by the latest Visual Studio 2017 update (15.5). Follow [these instructions](https://devtalk.nvidia.com/default/topic/1027209/cuda-setup-and-installation/cuda-9-0-does-not-work-with-the-latest-vs-2017-update/) to fix the issue.

## Usage
Program options:
 * `-f <path to model file>`: **(required)** A path to a polygon model file. Supported input formats: .ply, .off, .obj, .3DS, .SM and RAY.
 * `-s <voxel grid length>`: A power of 2, for the length of the cubical voxel grid. Default: 256, resulting in a 256 x 256 x 256 voxelization grid.  Cuda_voxelizer will automatically select the tightest bounding box around the model. 
 * `-o <output format>`:, The output format for voxelized models, currently *binvox* or *morton*. Default: *binvox*. The *morton* format is a tightly packed, morton-order representation. 

For example: `cuda_voxelizer -f bunny.ply -s 256` generates you a 256 x 256 x 256 bunny voxel model which will be stored in `bunny_256.binvox`. You can visualize this file using [viewvox](http://www.patrickmin.com/viewvox/).

![viewvox example](https://raw.githubusercontent.com/Forceflow/cuda_voxelizer/master/img/viewvox.JPG)

## Building
The project has the following build dependencies:
 * [Cuda 7.5 Toolkit (or higher)](https://developer.nvidia.com/cuda-toolkit) for CUDA.
 * [Trimesh2](https://github.com/Forceflow/trimesh2) for model importing. Latest version recommended.
 * [GLM](http://glm.g-truc.net/0.9.8/index.html) for vector math. Any recent version will do.

A Visual Studio 2017 project solution is provided in the `msvc`folder. It is configured for CUDA 9.0 RC, but you can edit the project file to make it work with lower CUDA versions. [Philipp-M](https://github.com/Philipp-M) was kind enough to write CMake support as well.

## Details
`cuda_voxelizer` implements an optimized version of the method described in M. Schwarz and HP Seidel's 2010 paper [*Fast Parallel Surface and Solid Voxelization on GPU's*](http://research.michael-schwarz.com/publ/2010/vox/). The morton-encoded table was based on my 2013 HPG paper [*Out-Of-Core construction of Sparse Voxel Octrees*](http://graphics.cs.kuleuven.be/publications/BLD14OCCSVO/)  and the work in [*libmorton*](https://github.com/Forceflow/libmorton).

`cuda_voxelizer` is built with a focus on performance. Usage of the routine as a per-frame voxelization step for real-time applications is viable. More performance metrics are on the todo list, but on a GTX 1060 these are the voxelization timings for the Stanford Bunny Model (1,55 MB, 70k triangles), including GPU memory transfers.

| Grid size | Time    |
|-----------|---------|
| 128^3     | 4.2 ms  |
| 256^3     | 6.2 ms  |
| 512^3     | 13.4 ms |
| 1024^3    | 38.6 ms  |

## Todo
 * Output to more popular voxel formats like MagicaVoxel, Minecraft
 * Optimize grid/block size launch parameters
 * Implement partitioning for larger models
 * Do a pre-pass to categorize triangles
 * Implement capture of normals / color / texture data
