[![Build Status](https://travis-ci.org/Forceflow/cuda_voxelizer.svg?branch=master)](https://travis-ci.org/Forceflow/cuda_voxelizer) ![](https://img.shields.io/github/license/Forceflow/cuda_voxelizer.svg) [![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.me/Forceflow)

# cuda_voxelizer v0.4.7
A command-line tool to convert polygon meshes to (annotated) voxel grids.
 * Supported input formats: .ply, .off, .obj, .3DS, .SM and RAY
 * Supported output formats: .binvox, .obj, morton ordered grid
 * Requires a CUDA-compatible video card. Compute Capability 2.0 or higher (Nvidia Fermi or better).
   * Since v0.4.4, the voxelizer reverts to a (slower) CPU voxelization method when no CUDA device is found

## Usage
Program options:
 * `-f <path to model file>`: **(required)** A path to a polygon-based 3D model file. 
 * `-s <voxel grid length>`: The length of the cubical voxel grid. Default: 256, resulting in a 256 x 256 x 256 voxelization grid.  The tool will automatically select the tightest cubical bounding box around the model.
 * `-o <output format>`: The output format for voxelized models, default: *binvox*. Output files are saved in the same folder as the input file.
   * `binvox`: A [binvox](http://www.patrickmin.com/binvox/binvox.html) file (default). Can be viewed using [viewvox](http://www.patrickmin.com/viewvox/).
   * `obj`: A mesh containing actual cubes (made up of triangle faces) for each voxel.
   * `obj_points`: A mesh containing a point cloud, with a vertex for each voxel. Can be viewed using any compatible viewer that can just display vertices, like [Blender](https://www.blender.org/) or [Meshlab](https://www.meshlab.net/).
   * `morton`: a binary file containing a Morton-ordered grid. This is a format I personally use for other tools.
 * `-cpu`: Force voxelization on the CPU instead of GPU. For when a CUDA device is not detected/compatible, or for very small models where GPU call overhead is not worth it. This is done multi-threaded, but can still be very slow for large models / grid sizes.
 * `-thrust` : Use Thrust library for copying the model data to the GPU, for a possible speed / throughput improvement. I found this to be very system-dependent. Default: disabled.
 * `-solid` : (Experimental) Use solid voxelization instead of voxelizing the mesh faces. Incompatible with `-cpu`. Needs a watertight input mesh.

  
## Examples

`cuda_voxelizer -f bunny.ply -s 256` generates a 256 x 256 x 256 binvox-based voxel model which will be stored in `bunny_256.binvox`. 

`cuda_voxelizer -f torus.ply -s 64 -o obj -thrust -solid` generates a solid (filled) 64 x 64 x 64 .obj voxel model which will be stored in `torus_64.obj`. During voxelization, the Cuda Thrust library will be used for a possible speedup, but YMMV.

![output_examples](https://raw.githubusercontent.com/Forceflow/cuda_voxelizer/master/img/output_examples.jpg)

## Building
The build process is aimed at 64-bit executables. It might be possible to build for 32-bit as well, but I'm not actively testing/supporting this.
### Dependencies
The project has the following build dependencies:
 * [Nvidia Cuda 8.0 Toolkit (or higher)](https://developer.nvidia.com/cuda-toolkit) for CUDA + Thrust libraries (standard included)
 * [Trimesh2](https://github.com/Forceflow/trimesh2) for model importing. Latest version recommended.
 * [GLM](http://glm.g-truc.net/0.9.8/index.html) for vector math. Any recent version will do.
 * [OpenMP](https://www.openmp.org/)

### Windows
A Visual Studio 2019 project solution is provided in the `msvc`folder. It is configured for CUDA 11, but you can edit the project file to make it work with lower CUDA versions. You can edit the `custom_includes.props` file to configure the library locations, and specify a place where the resulting binaries should be placed.

```
    <TRIMESH_DIR>C:\libs\trimesh2\</TRIMESH_DIR>
    <GLM_DIR>C:\libs\glm\</GLM_DIR>
    <BINARY_OUTPUT_DIR>D:\dev\Binaries\</BINARY_OUTPUT_DIR>
```

### Linux
[Philipp-M](https://github.com/Philipp-M) and [andreanicastro](https://github.com/andreanicastro) were kind enough to write [CMake](https://cmake.org/) support. Since November 2019, cuda_voxelizer also builds on [Travis CI](https://travis-ci.org/Forceflow/cuda_voxelizer), so check out the [yaml config file](https://github.com/Forceflow/cuda_voxelizer/blob/master/.travis.yml) for more Linux build support.

## Details
`cuda_voxelizer` implements an optimized version of the method described in M. Schwarz and HP Seidel's 2010 paper [*Fast Parallel Surface and Solid Voxelization on GPU's*](http://research.michael-schwarz.com/publ/2010/vox/). The morton-encoded table was based on my 2013 HPG paper [*Out-Of-Core construction of Sparse Voxel Octrees*](http://graphics.cs.kuleuven.be/publications/BLD14OCCSVO/)  and the work in [*libmorton*](https://github.com/Forceflow/libmorton).

`cuda_voxelizer` is built with a focus on performance. Usage of the routine as a per-frame voxelization step for real-time applications is viable. More performance metrics are on the todo list, but on a GTX 1060 these are the voxelization timings for the Stanford Bunny Model (1,55 MB, 70k triangles), including GPU memory transfers. Still lots of room for optimization.

| Grid size | Time    |
|-----------|---------|
| 128^3     | 4.2 ms  |
| 256^3     | 6.2 ms  |
| 512^3     | 13.4 ms |
| 1024^3    | 38.6 ms  |

## Notes / See Also
 * The .binvox file format was created by [Patrick Min](https://www.patrickmin.com/binvox/). Check some other interesting tools he wrote:
   * [viewvox](https://www.patrickmin.com/viewvox/): Visualization of voxel grids (a copy of this tool is included in cuda_voxelizer releases)
   * [thinvox](https://www.patrickmin.com/thinvox/): Thinning of voxel grids
 * Thanks to [conceptclear](https://github.com/conceptclear) for implementing solid voxelization
 * If you want a good customizable CPU-based voxelizer, I can recommend [VoxSurf](https://github.com/sylefeb/VoxSurf).
 * Another hackable voxel viewer is Sean Barrett's excellent [stb_voxel_render.h](https://github.com/nothings/stb/blob/master/stb_voxel_render.h).
 * Nvidia also has a voxel library called [GVDB](https://developer.nvidia.com/gvdb), that does a lot more than just voxelizing.

## Todo / Possible future work
This is on my list of nice things to add. Don't hesistate to crack one of these yourself and make a PR!

 * Noncubic grid support
 * Memory limits test
 * Output to more popular voxel formats like MagicaVoxel, Minecraft
 * Implement partitioning for larger models
 * Do a pre-pass to categorize triangles
 * Implement capture of normals / color / texture data
 
## Citation
If you use cuda_voxelizer in your published paper or other software, please reference it, for example as follows:
<pre>
@Misc{cudavoxelizer17,
author = "Jeroen Baert",
title = "Cuda Voxelizer: A GPU-accelerated Mesh Voxelizer",
howpublished = "\url{https://github.com/Forceflow/cuda_voxelizer}",
year = "2017"}
</pre>
If you end up using cuda_voxelizer in something cool, drop me an e-mail: **mail (at) jeroen-baert.be**
