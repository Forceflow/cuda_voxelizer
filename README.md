# cuda_voxelizer
Experimental CUDA voxelizer.
 * 
 * Outputs data to [.binvox file format](http://www.patrickmin.com/binvox/binvox.html) (default) or a morton-ordered grid.
 * Requirements to run: a CUDA-compatible video card, CM 2.0 or higher. The larger your models / voxel grid size, the more video memory will be required. Partitioning is still on the todo list.

## Usage
Program options:
 * `-f <path to model file>`: **(required)** A path to a polygon model file. Supported input formats: .ply, .off, .obj, .3DS, .SM and RAY.
 * `-s <voxel grid length>`: A power of 2, for the length of the cubical voxel grid. Default: 256, resulting in a 256 x 256 x 256 voxelization grid.
 * `-o <output format>`:, The output format for voxelized models, currently *binvox* or *morton*. Default: *binvox*. The *morton* format is a tightly packed, morton-order representation.

For example:

    cuda_voxelizer -f bunny.ply -s 256
    
generates you a 256x256x256 bunny voxel model.

## Building

