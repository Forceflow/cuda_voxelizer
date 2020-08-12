#pragma once

// GLM for maths
#include <TriMesh.h>
#include <glm/glm.hpp>
#include <cstdio>
#include <cmath> 
#include "util.h"
#include "timer.h"
#include "morton_LUTs.h"


namespace cpu_voxelizer {
	void cpu_voxelize_mesh(voxinfo info, trimesh::TriMesh* themesh, unsigned int* voxel_table, bool morton_order);
	void cpu_voxelize_mesh_solid(voxinfo info, trimesh::TriMesh* themesh, unsigned int* voxel_table, bool morton_order);
}