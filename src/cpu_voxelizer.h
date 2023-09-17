#pragma once

#include <TriMesh.h>
#include <cstdio>
#include <cmath> 
#include <omp.h>
#include "libs/cuda/helper_math.h"
#include "util.h"
#include "timer.h"
#include "morton_LUTs.h"

namespace cpu_voxelizer {
	void cpu_voxelize_mesh(voxinfo info, trimesh::TriMesh* themesh, unsigned int* voxel_table, bool morton_order);
	void cpu_voxelize_mesh_solid(voxinfo info, trimesh::TriMesh* themesh, unsigned int* voxel_table, bool morton_order);
}