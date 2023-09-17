![Build Status](https://github.com/Forceflow/cuda_voxelizer/actions/workflows/autobuild.yml/badge.svg) ![license](https://img.shields.io/github/license/Forceflow/cuda_voxelizer.svg)<br>
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Z8Z7GFNW3) 

# cuda_voxelizer v0.6

A command-line tool to convert polygon meshes to (annotated) voxel grids.
 * Supported input formats: .ply, .off, .obj, .3DS, .SM and RAY
 * Supported output formats: .vox, .binvox, .obj cubes and point cloud, morton ordered grid
 * Requires a CUDA-compatible video card. Compute Capability 2.0 or higher (Nvidia Fermi or better).
   * Since v0.4.4, the voxelizer reverts to a (slower) CPU voxelization method when no CUDA device is found
   
**Important:** _In v0.6 I replaced all GLM math types with builtin CUDA types, removing an external dependency. This is a big change. I've tested the release as well as I can, but if you encounter any weirdness, it's advised to check if you can reproduce the problem with an older version. Thanks!_

## Usage
Program options:
 * `-f <path to model file>`: **(required)** A path to a polygon-based 3D model file. 
 * `-s <voxel grid length>`: **(default: 256)** The length of the cubical voxel grid. The process will construct the tightest possible cubical bounding box around the input model.
 * `-o <output format>`: The output format for voxelized models, default: *binvox*. Output files are saved in the same folder as the input file, in the format `<original file name>_<grid_size>.extension`.
   * `vox`: **(default)** A [vox](https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt) file, which is the native format of and can be viewed with the excellent [MagicaVoxel](https://ephtracy.github.io/).
   * `binvox`: A [binvox](http://www.patrickmin.com/binvox/binvox.html) file. Can be viewed using [viewvox](http://www.patrickmin.com/viewvox/).
   * `obj`: A mesh containing actual cubes (made up of triangle faces) for each voxel.
   * `obj_points`: A mesh containing a point cloud, with a vertex for each voxel. Can be viewed using any compatible viewer that can just display vertices, like [Blender](https://www.blender.org/) or [Meshlab](https://www.meshlab.net/).
   * `morton`: a binary file containing a Morton-ordered grid. This is an internal format I use for other tools.
 * `-cpu`: Force multi-threaded voxelization on the CPU instead of GPU. Can be used when a CUDA device is not detected/compatible, or for very small models where GPU call overhead is not worth it.
 * `-solid` : (Experimental) Use solid voxelization instead of voxelizing the mesh faces. Needs a watertight input mesh.

## Examples
`cuda_voxelizer -f bunny.ply -s 256` generates a 256 x 256 x 256 vox-based voxel model which will be stored in `bunny_256.vox`. 

`cuda_voxelizer -f torus.ply -s 64 -o obj -solid` generates a solid (filled) 64 x 64 x 64 .obj voxel model which will be stored in `torus_64.obj`. 

![output_examples](https://raw.githubusercontent.com/Forceflow/cuda_voxelizer/main/img/output_examples.jpg)

## Building
The build process is aimed at 64-bit executables. It's possible to build for 32-bit as well, but I'm not actively testing/supporting this.
You can build using CMake or using the provided Visual Studio project. Since 2022, cuda_voxelizer builds via [Github Actions](https://github.com/Forceflow/cuda_voxelizer/actions) as well, check the .[yml config file](https://github.com/Forceflow/cuda_voxelizer/blob/main/.github/workflows/autobuild.yml) for more info.

### Dependencies
The project has the following build dependencies:
 * [Nvidia Cuda 8.0 Toolkit (or higher)](https://developer.nvidia.com/cuda-toolkit) for CUDA
 * [Trimesh2](https://github.com/Forceflow/trimesh2) for model importing. Latest version recommended.
 * [OpenMP](https://www.openmp.org/) for multi-threading.

### Build using CMake (Windows, Linux)
After installing dependencies, do `mkdir build` and `cd build`, followed by:

For Windows with Visual Studio:
```powershell
$env:CUDAARCHS="your_cuda_compute_capability"
cmake -A x64 -DTrimesh2_INCLUDE_DIR:PATH="path_to_trimesh2_include" -DTrimesh2_LINK_DIR:PATH="path_to_trimesh2_library_dir" .. 
```

For Linux:
```bash
CUDAARCHS="your_cuda_compute_capability" cmake -DTrimesh2_INCLUDE_DIR:PATH="path_to_trimesh2_include" -DTrimesh2_LINK_DIR:PATH="path_to_trimesh2_library_dir" -DCUDA_ARCH:STRING="your_cuda_compute_capability" .. 
```
Where `your_cuda_compute_capability` is a string specifying your CUDA architecture ([more info here](https://docs.nvidia.com/cuda/archive/10.2/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation-gpu-architecture) and [here CMake](https://cmake.org/cmake/help/v3.20/envvar/CUDAARCHS.html#envvar:CUDAARCHS)). For example: `CUDAARCHS="50;61"` or `CUDAARCHS="60"`.

Finally, run
```
cmake --build . --parallel number_of_cores
```

### Build using Visual Studio project (Windows)
A project solution for Visual Studio 2022 is provided in the `msvc` folder. It is configured for CUDA 12.1, but you can edit the project file to make it work with other CUDA versions. You can edit the `custom_includes.props` file to configure the library locations, and specify a place where the resulting binaries should be placed.

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

## Thanks
 * The [MagicaVoxel](https://ephtracy.github.io/) I/O was implemented using [MagicaVoxel File Writer](https://github.com/aiekick/MagicaVoxel_File_Writer) by [aiekick](https://github.com/aiekick).
* Thanks to [conceptclear](https://github.com/conceptclear) for implementing solid voxelization.

## See also

 * The [.binvox file format](https://www.patrickmin.com/binvox/binvox.html) was created by Michael Kazhdan. 
   * [Patrick Min](https://www.patrickmin.com/binvox/) wrote some interesting tools to work with it:
     * [viewvox](https://www.patrickmin.com/viewvox/): Visualization of voxel grids (a copy of this tool is included in cuda_voxelizer releases)
     * [thinvox](https://www.patrickmin.com/thinvox/): Thinning of voxel grids
   * [binvox-rw-py](https://github.com/dimatura/binvox-rw-py) is a Python module to interact with .binvox files
 * [Zarbuz](https://github.com/zarbuz)'s [FileToVox](https://github.com/Zarbuz/FileToVox) looks interesting as well
 * If you want a good customizable CPU-based voxelizer, I can recommend [VoxSurf](https://github.com/sylefeb/VoxSurf).
 * Another hackable voxel viewer is Sean Barrett's excellent [stb_voxel_render.h](https://github.com/nothings/stb/blob/master/stb_voxel_render.h).
 * Nvidia also has a voxel library called [GVDB](https://developer.nvidia.com/gvdb), that does a lot more than just voxelizing.

## Todo / Possible future work
This is on my list of "nice things to add".

 * Better output filename control
 * Noncubic grid support
 * Memory limits test
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

## Donate
cuda_voxelizer is developed in my free time. If you want to support the project, you can do so through:
* [Kofi](https://ko-fi.com/jbaert)
* BTC: 3GX3b7BZK2nhsneBG8eTqEchgCQ8FDfwZq 
* ETH: 0x7C9e97D2bBC2dFDd93EF56C77f626e802BA56860
