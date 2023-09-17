#include "cpu_voxelizer.h"
#define float_error 0.000001

namespace cpu_voxelizer {

	// Set specific bit in voxel table
	void setBit(unsigned int* voxel_table, size_t index) {
		size_t int_location = index / size_t(32);
		uint32_t bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
		uint32_t mask = 1 << bit_pos | 0;
		#pragma omp critical 
		{
			voxel_table[int_location] = (voxel_table[int_location] | mask);
		}
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
		Timer cpu_voxelization_timer; cpu_voxelization_timer.start();

		// PREPASS
		// Move all vertices to origin (can be done in parallel)
		trimesh::vec3 move_min = float3_to_trimesh<trimesh::vec3>(info.bbox.min);
#pragma omp parallel for
		for (int64_t i = 0; i < (int64_t) themesh->vertices.size(); i++) {
			if (i == 0) { printf("[Info] Using %d threads \n", omp_get_num_threads()); }
			themesh->vertices[i] = themesh->vertices[i] - move_min;
		}

#ifdef _DEBUG
		size_t debug_n_triangles = 0;
		size_t debug_n_voxels_tested = 0;
		size_t debug_n_voxels_marked = 0;
#endif

#pragma omp parallel for
		for (int64_t i = 0; i < (int64_t) info.n_triangles; i++) {
			// Common variables used in the voxelization process
			float3 delta_p = make_float3(info.unit.x, info.unit.y, info.unit.z);
			float3 c = make_float3(0.0f, 0.0f, 0.0f); // critical point
			int3 grid_max = make_int3(info.gridsize.x - 1, info.gridsize.y - 1, info.gridsize.z - 1); // grid max (grid runs from 0 to gridsize-1)
#ifdef _DEBUG
			debug_n_triangles++;
#endif
			// COMPUTE COMMON TRIANGLE PROPERTIES
			float3 v0 = trimesh_to_float3<trimesh::point>(themesh->vertices[themesh->faces[i][0]]);
			float3 v1 = trimesh_to_float3<trimesh::point>(themesh->vertices[themesh->faces[i][1]]);
			float3 v2 = trimesh_to_float3<trimesh::point>(themesh->vertices[themesh->faces[i][2]]);

			// Edge vectors
			float3 e0 = v1-v0;
			float3 e1 = v2-v1;
			float3 e2 = v0-v2;
			// Normal vector pointing up from the triangle
			float3 n = normalize(cross(e0, e1));

			// COMPUTE TRIANGLE BBOX IN GRID
			// Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
			AABox<float3> t_bbox_world(fminf(v0, fminf(v1, v2)), fmaxf(v0, fmaxf(v1, v2)));
			// Triangle bounding box in voxel grid coordinates is the world bounding box divided by the grid unit vector
			AABox<int3> t_bbox_grid;
			t_bbox_grid.min = clamp(float3_to_int3(t_bbox_world.min / info.unit), make_int3(0, 0, 0), grid_max);
			t_bbox_grid.max = clamp(float3_to_int3(t_bbox_world.max / info.unit), make_int3(0, 0, 0), grid_max);

			// PREPARE PLANE TEST PROPERTIES
			if (n.x > 0.0f) { c.x = info.unit.x; }
			if (n.y > 0.0f) { c.y = info.unit.y; }
			if (n.z > 0.0f) { c.z = info.unit.z; }
			float d1 = dot(n, (c - v0));
			float d2 = dot(n, ((delta_p - c) - v0));

			// PREPARE PROJECTION TEST PROPERTIES
			// XY plane
			float2 n_xy_e0 = make_float2(-1.0f * e0.y, e0.x);
			float2 n_xy_e1 = make_float2(-1.0f * e1.y, e1.x);
			float2 n_xy_e2 = make_float2(-1.0f * e2.y, e2.x);
			if (n.z < 0.0f) {
				n_xy_e0 = -n_xy_e0;
				n_xy_e1 = -n_xy_e1;
				n_xy_e2 = -n_xy_e2;
			}
			float d_xy_e0 = (-1.0f * dot(n_xy_e0, make_float2(v0.x, v0.y))) + max(0.0f, info.unit.x * n_xy_e0.x) + max(0.0f, info.unit.y * n_xy_e0.y);
			float d_xy_e1 = (-1.0f * dot(n_xy_e1, make_float2(v1.x, v1.y))) + max(0.0f, info.unit.x * n_xy_e1.x) + max(0.0f, info.unit.y * n_xy_e1.y);
			float d_xy_e2 = (-1.0f * dot(n_xy_e2, make_float2(v2.x, v2.y))) + max(0.0f, info.unit.x * n_xy_e2.x) + max(0.0f, info.unit.y * n_xy_e2.y);
			// YZ plane
			float2 n_yz_e0 = make_float2(-1.0f * e0.z, e0.y);
			float2 n_yz_e1 = make_float2(-1.0f * e1.z, e1.y);
			float2 n_yz_e2 = make_float2(-1.0f * e2.z, e2.y);
			if (n.x < 0.0f) {
				n_yz_e0 = -n_yz_e0;
				n_yz_e1 = -n_yz_e1;
				n_yz_e2 = -n_yz_e2;
			}
			float d_yz_e0 = (-1.0f * dot(n_yz_e0, make_float2(v0.y, v0.z))) + max(0.0f, info.unit.y * n_yz_e0.x) + max(0.0f, info.unit.z * n_yz_e0.y);
			float d_yz_e1 = (-1.0f * dot(n_yz_e1, make_float2(v1.y, v1.z))) + max(0.0f, info.unit.y * n_yz_e1.x) + max(0.0f, info.unit.z * n_yz_e1.y);
			float d_yz_e2 = (-1.0f * dot(n_yz_e2, make_float2(v2.y, v2.z))) + max(0.0f, info.unit.y * n_yz_e2.x) + max(0.0f, info.unit.z * n_yz_e2.y);
			// ZX plane
			float2 n_zx_e0 = make_float2(-1.0f * e0.x, e0.z);
			float2 n_zx_e1 = make_float2(-1.0f * e1.x, e1.z);
			float2 n_zx_e2 = make_float2(-1.0f * e2.x, e2.z);
			if (n.y < 0.0f) {
				n_zx_e0 = -n_zx_e0;
				n_zx_e1 = -n_zx_e1;
				n_zx_e2 = -n_zx_e2;
			}
			float d_xz_e0 = (-1.0f * dot(n_zx_e0, make_float2(v0.z, v0.x))) + max(0.0f, info.unit.x * n_zx_e0.x) + max(0.0f, info.unit.z * n_zx_e0.y);
			float d_xz_e1 = (-1.0f * dot(n_zx_e1, make_float2(v1.z, v1.x))) + max(0.0f, info.unit.x * n_zx_e1.x) + max(0.0f, info.unit.z * n_zx_e1.y);
			float d_xz_e2 = (-1.0f * dot(n_zx_e2, make_float2(v2.z, v2.x))) + max(0.0f, info.unit.x * n_zx_e2.x) + max(0.0f, info.unit.z * n_zx_e2.y);

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
						float3 p = make_float3(x * info.unit.x, y * info.unit.y, z * info.unit.z);
						float nDOTp = dot(n, p);
						if (((nDOTp + d1) * (nDOTp + d2)) > 0.0f) { continue; }

						// PROJECTION TESTS
						// XY
						float2 p_xy = make_float2(p.x, p.y);
						if ((dot(n_xy_e0, p_xy) + d_xy_e0) < 0.0f) { continue; }
						if ((dot(n_xy_e1, p_xy) + d_xy_e1) < 0.0f) { continue; }
						if ((dot(n_xy_e2, p_xy) + d_xy_e2) < 0.0f) { continue; }

						// YZ
						float2 p_yz = make_float2(p.y, p.z);
						if ((dot(n_yz_e0, p_yz) + d_yz_e0) < 0.0f) { continue; }
						if ((dot(n_yz_e1, p_yz) + d_yz_e1) < 0.0f) { continue; }
						if ((dot(n_yz_e2, p_yz) + d_yz_e2) < 0.0f) { continue; }

						// XZ	
						float2 p_zx = make_float2(p.z, p.x);
						if ((dot(n_zx_e0, p_zx) + d_xz_e0) < 0.0f) { continue; }
						if ((dot(n_zx_e1, p_zx) + d_xz_e1) < 0.0f) { continue; }
						if ((dot(n_zx_e2, p_zx) + d_xz_e2) < 0.0f) { continue; }
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
		cpu_voxelization_timer.stop(); std::fprintf(stdout, "[Perf] CPU voxelization time: %.1f ms \n", cpu_voxelization_timer.elapsed_time_milliseconds);
#ifdef _DEBUG
		printf("[Debug] Processed %llu triangles on the CPU \n", debug_n_triangles);
		printf("[Debug] Tested %llu voxels for overlap on CPU \n", debug_n_voxels_tested);
		printf("[Debug] Marked %llu voxels as filled (includes duplicates!) on CPU \n", debug_n_voxels_marked);
#endif
	}

	// use Xor for voxels whose corresponding bits have to flipped
	void setBitXor(unsigned int* voxel_table, size_t index) {
		size_t int_location = index / size_t(32);
		unsigned int bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
		unsigned int mask = 1 << bit_pos;
		#pragma omp critical 
		{
			voxel_table[int_location] = (voxel_table[int_location] ^ mask);
		}
	}

	bool TopLeftEdge(float2 v0, float2 v1) {
		return ((v1.y < v0.y) || (v1.y == v0.y && v0.x > v1.x));
	}

	//check the triangle is counterclockwise or not
	bool checkCCW(float2 v0, float2 v1, float2 v2) {
		float2 e0 = v1 - v0;
		float2 e1 = v2 - v0;
		float result = e0.x * e1.y - e1.x * e0.y;
		if (result > 0)
			return true;
		else
			return false;
	}

	//find the x coordinate of the voxel
	float get_x_coordinate(float3 n, float3 v0, float2 point) {
		return (-(n.y * (point.x - v0.y) + n.z * (point.y - v0.z)) / n.x + v0.x);
	}

	//check the location with point and triangle
	int check_point_triangle(float2 v0, float2 v1, float2 v2, float2 point) {
		float2 PA = point - v0;
		float2 PB = point - v1;
		float2 PC = point - v2;

		float t1 = PA.x * PB.y - PA.y * PB.x;
		if (std::fabs(t1) < float_error && PA.x * PB.x <= 0 && PA.y * PB.y <= 0)
			return 1;

		float t2 = PB.x * PC.y - PB.y * PC.x;
		if (std::fabs(t2) < float_error && PB.x * PC.x <= 0 && PB.y * PC.y <= 0)
			return 2;

		float t3 = PC.x * PA.y - PC.y * PA.x;
		if (std::fabs(t3) < float_error && PC.x * PA.x <= 0 && PC.y * PA.y <= 0)
			return 3;

		if (t1 * t2 > 0 && t1 * t3 > 0)
			return 0;
		else
			return -1;
	}

	// Mesh voxelization method
	void cpu_voxelize_mesh_solid(voxinfo info, trimesh::TriMesh* themesh, unsigned int* voxel_table, bool morton_order) {
		Timer cpu_voxelization_timer; cpu_voxelization_timer.start();

		// PREPASS
		// Move all vertices to origin (can be done in parallel)
		trimesh::vec3 move_min = float3_to_trimesh<trimesh::vec3>(info.bbox.min);
#pragma omp parallel for
		for (int64_t i = 0; i < (int64_t) themesh->vertices.size(); i++) {
			if (i == 0) { printf("[Info] Using %d threads \n", omp_get_num_threads()); }
			themesh->vertices[i] = themesh->vertices[i] - move_min;
		}

#pragma omp parallel for
		for (int64_t i = 0; i < (int64_t) info.n_triangles; i++) {
			// Triangle vertices
			float3 v0 = trimesh_to_float3<trimesh::point>(themesh->vertices[themesh->faces[i][0]]);
			float3 v1 = trimesh_to_float3<trimesh::point>(themesh->vertices[themesh->faces[i][1]]);
			float3 v2 = trimesh_to_float3<trimesh::point>(themesh->vertices[themesh->faces[i][2]]);
			// Edge vectors
			float3 e0 = v1 - v0;
			float3 e1 = v2 - v1;
			float3 e2 = v0 - v2;
			// Normal vector pointing up from the triangle
			float3 n = normalize(cross(e0, e1));
			if (std::fabs(n.x) < float_error) {continue;}

			// Calculate the projection of three point into yoz plane
			float2 v0_yz = make_float2(v0.y, v0.z);
			float2 v1_yz = make_float2(v1.y, v1.z);
			float2 v2_yz = make_float2(v2.y, v2.z);

			// Set the triangle counterclockwise
			if (!checkCCW(v0_yz, v1_yz, v2_yz))
			{
				float2 v3 = v1_yz;
				v1_yz = v2_yz;
				v2_yz = v3;
			}

			// COMPUTE TRIANGLE BBOX IN GRID
			// Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
			float2 bbox_max = fmaxf(v0_yz, fmaxf(v1_yz, v2_yz));
			float2 bbox_min = fminf(v0_yz, fminf(v1_yz, v2_yz));

			float2 bbox_max_grid = make_float2(floor(bbox_max.x / info.unit.y - 0.5f), floor(bbox_max.y / info.unit.z - 0.5f));
			float2 bbox_min_grid = make_float2(ceil(bbox_min.x / info.unit.y - 0.5f), ceil(bbox_min.y / info.unit.z - 0.5f));

			for (int y = static_cast<int>(bbox_min_grid.x); y <= bbox_max_grid.x; y++)
			{
				for (int z = static_cast<int>(bbox_min_grid.y); z <= bbox_max_grid.y; z++)
				{
					float2 point = make_float2((y + 0.5f) * info.unit.y, (z + 0.5f) * info.unit.z);
					int checknum = check_point_triangle(v0_yz, v1_yz, v2_yz, point);
					if ((checknum == 1 && TopLeftEdge(v0_yz, v1_yz)) || (checknum == 2 && TopLeftEdge(v1_yz, v2_yz)) || (checknum == 3 && TopLeftEdge(v2_yz, v0_yz)) || (checknum == 0))
					{
						unsigned int xmax = int(get_x_coordinate(n, v0, point) / info.unit.x - 0.5);
						for (unsigned int x = 0; x <= xmax; x++)
						{
							if (morton_order) {
								size_t location = mortonEncode_LUT(x, y, z);
								setBitXor(voxel_table, location);
							}
							else {
								size_t location = static_cast<size_t>(x) + (static_cast<size_t>(y) * static_cast<size_t>(info.gridsize.y)) + (static_cast<size_t>(z) * static_cast<size_t>(info.gridsize.y) * static_cast<size_t>(info.gridsize.z));
								setBitXor(voxel_table, location);
							}
							continue;
						}
					}
				}
			}
		}
		cpu_voxelization_timer.stop(); fprintf(stdout, "[Perf] CPU voxelization time: %.1f ms \n", cpu_voxelization_timer.elapsed_time_milliseconds);
	}
}