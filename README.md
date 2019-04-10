![](https://img.shields.io/github/license/Forceflow/cuda_voxelizer.svg) [![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=4JAUNWWYUVRN4&currency_code=EUR&source=url)

# cuda_voxelizer v0.3
Experimental CUDA voxelizer, to convert polygon meshes to (annotated) voxel grids. 
 * Outputs data to [.binvox file format](http://www.patrickmin.com/binvox/binvox.html) (default) or a morton-ordered grid. More output formats (magicavoxel, minecraft schematic) are in development.
 * Requires a CUDA-compatible video card. Compute Capability 2.0 or higher (Nvidia Fermi or better).
 * Current project targets CUDA 10.1
 * 64-bit executables only. 32-bit might work, but you're on your own :)

## Usage
Program options:
 * `-f <path to model file>`: **(required)** A path to a polygon model file. Supported input formats: .ply, .off, .obj, .3DS, .SM and RAY.
 * `-s <voxel grid length>`: The length of the cubical voxel grid. Default: 256, resulting in a 256 x 256 x 256 voxelization grid.  Cuda_voxelizer will automatically select the tightest bounding box around the model. If you want to output to the *morton* file format, this should be a power of 2.
 * `-o <output format>`:, The output format for voxelized models, currently *binvox* or *morton*. Default: *binvox*. The *morton* format is a tightly packed, morton-order representation. 
  * `-t` : Use Thrust library for CUDA memory operations. Might provide speed / throughput improvement. Default: disabled.

For example: `cuda_voxelizer -f bunny.ply -s 256` generates you a 256 x 256 x 256 bunny voxel model which will be stored in `bunny_256.binvox`. You can visualize this file using [viewvox](http://www.patrickmin.com/viewvox/).

![viewvox example](https://raw.githubusercontent.com/Forceflow/cuda_voxelizer/master/img/viewvox.JPG)

## Building
The project has the following build dependencies:
 * [Cuda 8.0 Toolkit (or higher)](https://developer.nvidia.com/cuda-toolkit) for CUDA.
 * Cuda Thrust libraries (they come with the toolkit).
 * [Trimesh2](https://github.com/Forceflow/trimesh2) for model importing. Latest version recommended.
 * [GLM](http://glm.g-truc.net/0.9.8/index.html) for vector math. Any recent version will do.

A Visual Studio 2017 project solution is provided in the `msvc`folder. It is configured for CUDA 10.0, but you can edit the project file to make it work with lower CUDA versions. You can edit the `custom_includes.props` file to configure the library locations, and specify a place where the resulting binaries should be placed.

```
    <TRIMESH_DIR>C:\libs\trimesh2\</TRIMESH_DIR>
    <GLM_DIR>C:\libs\glm\</GLM_DIR>
    <BINARY_OUTPUT_DIR>D:\dev\Binaries\</BINARY_OUTPUT_DIR>
```

[Philipp-M](https://github.com/Philipp-M) and [andreanicastro](https://github.com/andreanicastro) were kind enough to write CMake support as well.

## Details
`cuda_voxelizer` implements an optimized version of the method described in M. Schwarz and HP Seidel's 2010 paper [*Fast Parallel Surface and Solid Voxelization on GPU's*](http://research.michael-schwarz.com/publ/2010/vox/). The morton-encoded table was based on my 2013 HPG paper [*Out-Of-Core construction of Sparse Voxel Octrees*](http://graphics.cs.kuleuven.be/publications/BLD14OCCSVO/)  and the work in [*libmorton*](https://github.com/Forceflow/libmorton).

`cuda_voxelizer` is built with a focus on performance. Usage of the routine as a per-frame voxelization step for real-time applications is viable. More performance metrics are on the todo list, but on a GTX 1060 these are the voxelization timings for the Stanford Bunny Model (1,55 MB, 70k triangles), including GPU memory transfers. Still lots of room for optimization.

| Grid size | Time    |
|-----------|---------|
| 128^3     | 4.2 ms  |
| 256^3     | 6.2 ms  |
| 512^3     | 13.4 ms |
| 1024^3    | 38.6 ms  |

## Notes
 * If you want a good customizable CPU-based voxelizer, I can recommend [VoxSurf](https://github.com/sylefeb/VoxSurf).
 * Another hackable voxel viewer is Sean Barrett's excellent [stb_voxel_render.h](https://github.com/nothings/stb/blob/master/stb_voxel_render.h).
 * Nvidia also has a voxel library called [GVDB](https://developer.nvidia.com/gvdb), that does a lot more than just voxelizing.

## Todo
 * Parallelize / CUDA-ify bounding box estimation
 * Output to more popular voxel formats like MagicaVoxel, Minecraft
 * Optimize grid/block size launch parameters
 * Implement partitioning for larger models
 * Do a pre-pass to categorize triangles
 * Implement capture of normals / color / texture data

