#include "cpu_voxelizer.h"

namespace cpu_voxelizer {

	// Set specific bit in voxel table
	void setBit(unsigned int* voxel_table, size_t index) {
		size_t int_location = index / size_t(32);
		uint32_t bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
		uint32_t mask = 1 << bit_pos | 0;
		voxel_table[int_location] = (voxel_table[int_location] | mask);
	}

	// Encode morton code using LUT table
	uint64_t mortonEncode_LUT(unsigned int x, unsigned int y, unsigned int z) {
		uint64_t answer = 0;
		answer = host_morton256_z[(z >> 16) & 0xFF] |
			host_morton256_y[(y >> 16) & 0xFF] |
			host_morton256_x[(x >> 16) & 0xFF];
		answer = answer << 48 |
			host_morton256_z[(z >> 8) & 0xFF] |
			host_morton256_y[(y >> 8) & 0xFF] |
			host_morton256_x[(x >> 8) & 0xFF];
		answer = answer << 24 |
			host_morton256_z[(z) & 0xFF] |
			host_morton256_y[(y) & 0xFF] |
			host_morton256_x[(x) & 0xFF];
		return answer;
	}

	// Mesh voxelization method
	void cpu_voxelize_mesh(voxinfo info, trimesh::TriMesh* themesh, unsigned int* voxel_table, bool morton_order) {
		//// Common variables used in the voxelization process
		//glm::vec3 delta_p(info.unit.x, info.unit.y, info.unit.z);
		//glm::vec3 c(0.0f, 0.0f, 0.0f); // critical point
		//glm::vec3 grid_max(info.gridsize.x - 1, info.gridsize.y - 1, info.gridsize.z - 1); // grid max (grid runs from 0 to gridsize-1)


		// PREPASS
		// Move all vertices to origin (can be done in parallel)
		trimesh::vec3 move_min = glm_to_trimesh<trimesh::vec3>(info.bbox.min);
#pragma omp for
		for (uint64_t i = 0; i < themesh->vertices.size(); i++) {
			themesh->vertices[i] = themesh->vertices[i] - move_min;
		}

#ifdef _DEBUG
		size_t debug_n_triangles = 0;
		size_t debug_n_voxels_tested = 0;
		size_t debug_n_voxels_marked = 0;
#endif

		for (size_t i = 0; i < info.n_triangles; i++) {
			// Common variables used in the voxelization process
			glm::vec3 delta_p(info.unit.x, info.unit.y, info.unit.z);
			glm::vec3 c(0.0f, 0.0f, 0.0f); // critical point
			glm::vec3 grid_max(info.gridsize.x - 1, info.gridsize.y - 1, info.gridsize.z - 1); // grid max (grid runs from 0 to gridsize-1)
#ifdef _DEBUG
			debug_n_triangles++;
#endif
			// COMPUTE COMMON TRIANGLE PROPERTIES
			// Move vertices to origin using bbox
			glm::vec3 v0 = trimesh_to_glm<trimesh::point>(themesh->vertices[themesh->faces[i][0]]);
			glm::vec3 v1 = trimesh_to_glm<trimesh::point>(themesh->vertices[themesh->faces[i][1]]);
			glm::vec3 v2 = trimesh_to_glm<trimesh::point>(themesh->vertices[themesh->faces[i][2]]);

			// Edge vectors
			glm::vec3 e0 = v1 - v0;
			glm::vec3 e1 = v2 - v1;
			glm::vec3 e2 = v0 - v2;
			// Normal vector pointing up from the triangle
			glm::vec3 n = glm::normalize(glm::cross(e0, e1));

			// COMPUTE TRIANGLE BBOX IN GRID
			// Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
			AABox<glm::vec3> t_bbox_world(glm::min(v0, glm::min(v1, v2)), glm::max(v0, glm::max(v1, v2)));
			// Triangle bounding box in voxel grid coordinates is the world bounding box divided by the grid unit vector
			AABox<glm::ivec3> t_bbox_grid;
			t_bbox_grid.min = glm::clamp(t_bbox_world.min / info.unit, glm::vec3(0.0f, 0.0f, 0.0f), grid_max);
			t_bbox_grid.max = glm::clamp(t_bbox_world.max / info.unit, glm::vec3(0.0f, 0.0f, 0.0f), grid_max);

			// PREPARE PLANE TEST PROPERTIES
			if (n.x > 0.0f) { c.x = info.unit.x; }
			if (n.y > 0.0f) { c.y = info.unit.y; }
			if (n.z > 0.0f) { c.z = info.unit.z; }
			float d1 = glm::dot(n, (c - v0));
			float d2 = glm::dot(n, ((delta_p - c) - v0));

			// PREPARE PROJECTION TEST PROPERTIES
			// XY plane
			glm::vec2 n_xy_e0(-1.0f * e0.y, e0.x);
			glm::vec2 n_xy_e1(-1.0f * e1.y, e1.x);
			glm::vec2 n_xy_e2(-1.0f * e2.y, e2.x);
			if (n.z < 0.0f) {
				n_xy_e0 = -n_xy_e0;
				n_xy_e1 = -n_xy_e1;
				n_xy_e2 = -n_xy_e2;
			}
			float d_xy_e0 = (-1.0f * glm::dot(n_xy_e0, glm::vec2(v0.x, v0.y))) + glm::max(0.0f, info.unit.x * n_xy_e0[0]) + glm::max(0.0f, info.unit.y * n_xy_e0[1]);
			float d_xy_e1 = (-1.0f * glm::dot(n_xy_e1, glm::vec2(v1.x, v1.y))) + glm::max(0.0f, info.unit.x * n_xy_e1[0]) + glm::max(0.0f, info.unit.y * n_xy_e1[1]);
			float d_xy_e2 = (-1.0f * glm::dot(n_xy_e2, glm::vec2(v2.x, v2.y))) + glm::max(0.0f, info.unit.x * n_xy_e2[0]) + glm::max(0.0f, info.unit.y * n_xy_e2[1]);
			// YZ plane
			glm::vec2 n_yz_e0(-1.0f * e0.z, e0.y);
			glm::vec2 n_yz_e1(-1.0f * e1.z, e1.y);
			glm::vec2 n_yz_e2(-1.0f * e2.z, e2.y);
			if (n.x < 0.0f) {
				n_yz_e0 = -n_yz_e0;
				n_yz_e1 = -n_yz_e1;
				n_yz_e2 = -n_yz_e2;
			}
			float d_yz_e0 = (-1.0f * glm::dot(n_yz_e0, glm::vec2(v0.y, v0.z))) + glm::max(0.0f, info.unit.y * n_yz_e0[0]) + glm::max(0.0f, info.unit.z * n_yz_e0[1]);
			float d_yz_e1 = (-1.0f * glm::dot(n_yz_e1, glm::vec2(v1.y, v1.z))) + glm::max(0.0f, info.unit.y * n_yz_e1[0]) + glm::max(0.0f, info.unit.z * n_yz_e1[1]);
			float d_yz_e2 = (-1.0f * glm::dot(n_yz_e2, glm::vec2(v2.y, v2.z))) + glm::max(0.0f, info.unit.y * n_yz_e2[0]) + glm::max(0.0f, info.unit.z * n_yz_e2[1]);
			// ZX plane
			glm::vec2 n_zx_e0(-1.0f * e0.x, e0.z);
			glm::vec2 n_zx_e1(-1.0f * e1.x, e1.z);
			glm::vec2 n_zx_e2(-1.0f * e2.x, e2.z);
			if (n.y < 0.0f) {
				n_zx_e0 = -n_zx_e0;
				n_zx_e1 = -n_zx_e1;
				n_zx_e2 = -n_zx_e2;
			}
			float d_xz_e0 = (-1.0f * glm::dot(n_zx_e0, glm::vec2(v0.z, v0.x))) + glm::max(0.0f, info.unit.x * n_zx_e0[0]) + glm::max(0.0f, info.unit.z * n_zx_e0[1]);
			float d_xz_e1 = (-1.0f * glm::dot(n_zx_e1, glm::vec2(v1.z, v1.x))) + glm::max(0.0f, info.unit.x * n_zx_e1[0]) + glm::max(0.0f, info.unit.z * n_zx_e1[1]);
			float d_xz_e2 = (-1.0f * glm::dot(n_zx_e2, glm::vec2(v2.z, v2.x))) + glm::max(0.0f, info.unit.x * n_zx_e2[0]) + glm::max(0.0f, info.unit.z * n_zx_e2[1]);

			// test possible grid boxes for overlap
			for (int z = t_bbox_grid.min.z; z <= t_bbox_grid.max.z; z++) {
				for (int y = t_bbox_grid.min.y; y <= t_bbox_grid.max.y; y++) {
					for (int x = t_bbox_grid.min.x; x <= t_bbox_grid.max.x; x++) {
						// size_t location = x + (y*info.gridsize) + (z*info.gridsize*info.gridsize);
						// if (checkBit(voxel_table, location)){ continue; }
#ifdef _DEBUG
						debug_n_voxels_tested++;
#endif

						// TRIANGLE PLANE THROUGH BOX TEST
						glm::vec3 p(x * info.unit.x, y * info.unit.y, z * info.unit.z);
						float nDOTp = glm::dot(n, p);
						if (((nDOTp + d1) * (nDOTp + d2)) > 0.0f) { continue; }

						// PROJECTION TESTS
						// XY
						glm::vec2 p_xy(p.x, p.y);
						if ((glm::dot(n_xy_e0, p_xy) + d_xy_e0) < 0.0f) { continue; }
						if ((glm::dot(n_xy_e1, p_xy) + d_xy_e1) < 0.0f) { continue; }
						if ((glm::dot(n_xy_e2, p_xy) + d_xy_e2) < 0.0f) { continue; }

						// YZ
						glm::vec2 p_yz(p.y, p.z);
						if ((glm::dot(n_yz_e0, p_yz) + d_yz_e0) < 0.0f) { continue; }
						if ((glm::dot(n_yz_e1, p_yz) + d_yz_e1) < 0.0f) { continue; }
						if ((glm::dot(n_yz_e2, p_yz) + d_yz_e2) < 0.0f) { continue; }

						// XZ	
						glm::vec2 p_zx(p.z, p.x);
						if ((glm::dot(n_zx_e0, p_zx) + d_xz_e0) < 0.0f) { continue; }
						if ((glm::dot(n_zx_e1, p_zx) + d_xz_e1) < 0.0f) { continue; }
						if ((glm::dot(n_zx_e2, p_zx) + d_xz_e2) < 0.0f) { continue; }
#ifdef _DEBUG
						debug_n_voxels_marked += 1;
#endif
						if (morton_order) {
							size_t location = mortonEncode_LUT(x, y, z);
							setBit(voxel_table, location);
						}
						else {
							size_t location = static_cast<size_t>(x) + (static_cast<size_t>(y)* static_cast<size_t>(info.gridsize.y)) + (static_cast<size_t>(z)* static_cast<size_t>(info.gridsize.y)* static_cast<size_t>(info.gridsize.z));
							//std:: cout << "Voxel found at " << x << " " << y << " " << z << std::endl;
							setBit(voxel_table, location);
						}
						continue;
					}
				}
			}
		}
#ifdef _DEBUG
		printf("[Debug] Processed %llu triangles on the CPU \n", debug_n_triangles);
		printf("[Debug] Tested %llu voxels for overlap on CPU \n", debug_n_voxels_tested);
		printf("[Debug] Marked %llu voxels as filled (includes duplicates!) on CPU \n", debug_n_voxels_marked);
#endif
	}
}