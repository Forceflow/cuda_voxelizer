cuda_voxelizer
==============

Experimental CUDA voxelizer. Outputs data to .binvox file format (default) or a morton-ordered grid (using the switch -m). For model importing, this tool uses [Trimesh2](https://github.com/Forceflow/trimesh2), so all usual file formats are supported (.ply, .obj, ...).

In order for this tool to work, you need a CUDA-compatible video card, with at least compute model 2.0 supported. Any Nvidia card less than 5 years old will probably do.

Usage:

    cuda_voxelizer -f (path to filename) -s (size, power of 2)

So

    cuda_voxelizer -f bunny.ply -s 256

generates you a 256x256x256 bunny voxel model.
