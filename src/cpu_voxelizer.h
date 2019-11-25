#pragma once

// GLM for maths
#include <TriMesh.h>
#include <glm/glm.hpp>
#include "util.h"
#include "morton_LUTs.h"
#include <cstdio>

namespace cpu_voxelizer {
	void cpu_voxelize_mesh(voxinfo info, trimesh::TriMesh* themesh, unsigned int* voxel_table, bool morton_order);
}