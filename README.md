# cuda_voxelizer v0.1
Experimental CUDA voxelizer, to convert polygon meshes to annotated voxel grids. 
 * Outputs data to [.binvox file format](http://www.patrickmin.com/binvox/binvox.html) (default) or a morton-ordered grid. More output formats (magicavoxel, minecraft schematic) are in development.
 * Requires a CUDA-compatible video card. Compute Capability 2.0 or higher (Nvidia Fermi or better).
## Usage
Program options:
 * `-f <path to model file>`: **(required)** A path to a polygon model file. Supported input formats: .ply, .off, .obj, .3DS, .SM and RAY.
 * `-s <voxel grid length>`: A power of 2, for the length of the cubical voxel grid. Default: 256, resulting in a 256 x 256 x 256 voxelization grid.  Cuda_voxelizer will automatically select the tightest bounding box around the model. 
 * `-o <output format>`:, The output format for voxelized models, currently *binvox* or *morton*. Default: *binvox*. The *morton* format is a tightly packed, morton-order representation. 

For example: `cuda_voxelizer -f bunny.ply -s 256` generates you a 256 x 256 x 256 bunny voxel model which will be stored in `bunny_256.binvox`. You can visualize this file using [viewvox](http://www.patrickmin.com/viewvox/).

![viewvox example](https://raw.githubusercontent.com/Forceflow/cuda_voxelizer/master/img/viewvox.JPG)

## Building
### Dependencies
 * [Cuda 7.5 Toolkit](https://developer.nvidia.com/cuda-toolkit) for CUDA.
 * [Trimesh2](https://github.com/Forceflow/trimesh2) for model importing.
 * [GLM](http://glm.g-truc.net/0.9.8/index.html) for vector math.

A Visual studio 