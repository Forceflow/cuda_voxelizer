cuda_voxelizer
==============
Experimental CUDA voxelizer.
 * Supported input files: .ply, .off, .obj, .3DS, .SM, RAY.
 * Outputs data to [.binvox file format](http://www.patrickmin.com/binvox/binvox.html) (default) or a morton-ordered grid (using the switch -m).
 * Requirements to run: a CUDA-compatible video card, CM 2.0 or higher. The larger your models / voxel grid size, the more video memory will be required. Partitioning is still on the todo list.

## Usage
Usage:

    cuda_voxelizer -f (path to filename) -s (size, power of 2)
    
For example:

    cuda_voxelizer -f bunny.ply -s 256
    
generates you a 256x256x256 bunny voxel model.

## Building

