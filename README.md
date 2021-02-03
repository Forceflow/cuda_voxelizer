[![Build Status](https://travis-ci.org/Forceflow/cuda_voxelizer.svg?branch=master)](https://travis-ci.org/Forceflow/cuda_voxelizer) ![](https://img.shields.io/github/license/Forceflow/cuda_voxelizer.svg) [![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.me/Forceflow)

# cuda_voxelizer v0.4.10
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
 * `-cpu`: Force voxelization on the CPU instead of GPU. For when a CUDA device is not detected/compatible, or for very small models where GPU call overhead is not worth it. This is done multi-threaded, but will be slower for large models / grid sizes.
 * `-thrust` : Use Thrust library for copying the model data to the GPU, for a possible speed / throughput improvement. I found this to be very system-dependent. Default: disabled.
 * `-solid` : (Experimental) Use solid voxelization instead of voxelizing the mesh faces. Needs a watertight input mesh.

  
## Examples

`cuda_voxelizer -f bunny.ply -s 256` generates a 256 x 256 x 256 binvox-based voxel model which will be stored in `bunny_256.binvox`. 

`cuda_voxelizer -f torus.ply -s 64 -o obj -thrust -solid` generates a solid (filled) 64 x 64 x 64 .obj voxel model which will be stored in `torus_64.obj`. During voxelization, the Cuda Thrust library will be used for a possible speedup, but YMMV.

![output_examples](https://raw.githubusercontent.com/Forceflow/cuda_voxelizer/main/img/output_examples.jpg)

## Building
The build process is aimed at 64-bit executables. It might be possible to build for 32-bit as well, but I'm not actively testing/supporting this.
You can build using CMake, or using the provided Visual Studio project. Since November 2019, cuda_voxelizer also builds on [Travis CI](https://travis-ci.org/Forceflow/cuda_voxelizer), so check out the [yaml config file](https://github.com/Forceflow/cuda_voxelizer/blob/main/.travis.yml) for more Linux build support.

### Dependencies
The project has the following build dependencies:
 * [Nvidia Cuda 8.0 Toolkit (or higher)](https://developer.nvidia.com/cuda-toolkit) for CUDA + Thrust libraries (standard included)
 * [Trimesh2](https://github.com/Forceflow/trimesh2) for model importing. Latest version recommended.
 * [GLM](http://glm.g-truc.net/0.9.8/index.html) for vector math. Any recent version will do.
 * [OpenMP](https://www.openmp.org/)

### Build using CMake (Windows, Linux)

After installing dependencies, do `mkdir build` and `cd build`, followed by:

For Windows with Visual Studio 2019:
```
cmake -A x64 -DTrimesh2_INCLUDE_DIR:PATH="path_to_trimesh2_include" -DTrimesh2_LINK_DIR:PATH="path_to_trimesh2_library_dir" -DCUDA_ARCH:STRING="your_cuda_compute_capability" .. 
```

For Linux:
```
cmake -DTrimesh2_INCLUDE_DIR:PATH="path_to_trimesh2_include" -DTrimesh2_LINK_DIR:PATH="path_to_trimesh2_library_dir" -DCUDA_ARCH:STRING="your_cuda_compute_capability" .. 
```
Where `your_cuda_compute_capability` is a string specifying your CUDA architecture ([more info here](https://docs.nvidia.com/cuda/archive/10.2/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation-gpu-architecture)). For example: `-DCUDA_ARCH:STRING=61` or `-DCUDA_ARCH:STRING=60`.

Finally, run
```
cmake --build . -j number_of_cores
```

### Build using Visual Studio project (Windows)

A Visual Studio 2019 project solution is provided in the `msvc`folder. It is configured for CUDA 11, but you can edit the project file to make it work with lower CUDA versions. You can edit the `custom_includes.props` file to configure the library locations, and specify a place where the resulting binaries should be placed.

```
    <TRIMESH_DIR>C:\libs\trimesh2\</TRIMESH_DIR>
    <GLM_DIR>C:\libs\glm\</GLM_DIR>
    <BINARY_OUTPUT_DIR>D:\dev\Binaries\</BINARY_OUTPUT_DIR>
```
## Details
`cuda_voxelizer` implements an optimized version of the method described in M. Schwarz and HP Seidel's 2010 paper [*Fast Parallel Surface and Solid Voxelization on GPU's*](http://research.michael-schwarz.com/publ/2010/vox/). The morton-encoded table was based on my 2013 HPG paper [*Out-Of-Core construction of Sparse Voxel Octrees*](http://graphics.cs.kuleuven.be/publications/BLD14OCCSVO/)  and the work in [*libmorton*](https://github.com/Forceflow/libmorton).

`cuda_voxelizer` is built with a focus on performance. Usage of the routine as a per-frame voxelization step for real-time applications is viable. These are the voxelization timings for the [Stanford Bunny Model](https://graphics.stanford.edu/data/3Dscanrep/) (1,55 MB, 70k triangles). 
 * This is the voxelization time for a non-solid voxelization. No I/O - from disk or to GPU - is included in this timing.
 * CPU voxelization time is heavily dependent on how many cores your CPU has - OpenMP allocates 1 thread per core.

| Grid size | GPU (GTX 1050 TI) | CPU (Intel i7 8750H, 12 threads) |
|-----------|--------|--------|
| 64³     | 0.2 ms | 39.8 ms |
| 128³     | 0.3 ms | 63.6 ms |
| 256³     | 0.6 ms | 118.2 ms |
| 512³     | 1.8 ms | 308.8 ms |
| 1024³    | 8.6 ms | 1047.5 ms |
| 2048³    | 44.6 ms | 4147.4 ms |

## Notes / See Also
 * The [.binvox file format](https://www.patrickmin.com/binvox/binvox.html) was created by Michael Kazhdan. 
   * [Patrick Min](https://www.patrickmin.com/binvox/) wrote some interesting tools to work with it:
     * [viewvox](https://www.patrickmin.com/viewvox/): Visualization of voxel grids (a copy of this tool is included in cuda_voxelizer releases)
     * [thinvox](https://www.patrickmin.com/thinvox/): Thinning of voxel grids
   * [binvox-rw-py](https://github.com/dimatura/binvox-rw-py) is a Python module to interact with .binvox files
 * Thanks to [conceptclear](https://github.com/conceptclear) for implementing solid voxelization
 * [Zarbuz](https://github.com/zarbuz)'s [FileToVox](https://github.com/Zarbuz/FileToVox) looks interesting as well
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
