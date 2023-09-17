#include "voxelize.cuh"

// CUDA Global Memory variables

// Debug counters for some sanity checks
#ifdef _DEBUG
__device__ size_t debug_d_n_voxels_marked = 0;
__device__ size_t debug_d_n_triangles = 0;
__device__ size_t debug_d_n_voxels_tested = 0;
#endif

// Possible optimization: buffer bitsets (for now: Disabled because too much overhead)
//struct bufferedBitSetter{
//	unsigned int* voxel_table;
//	size_t current_int_location;
//	unsigned int current_mask;
//
//	__device__ __inline__ bufferedBitSetter(unsigned int* voxel_table, size_t index) :
//		voxel_table(voxel_table), current_mask(0) {
//		current_int_location = int(index / 32.0f);
//	}
//
//	__device__ __inline__ void setBit(size_t index){
//		size_t new_int_location = int(index / 32.0f);
//		if (current_int_location != new_int_location){
//			flush();
//			current_int_location = new_int_location;
//		}
//		unsigned int bit_pos = 31 - (unsigned int)(int(index) % 32);
//		current_mask = current_mask | (1 << bit_pos);
//	}
//
//	__device__ __inline__ void flush(){
//		if (current_mask != 0){
//			atomicOr(&(voxel_table[current_int_location]), current_mask);
//		}
//	}
//};

// Possible optimization: check bit before you set it - don't need to do atomic operation if it's already set to 1
// For now: overhead, so it seems
//__device__ __inline__ bool checkBit(unsigned int* voxel_table, size_t index){
//	size_t int_location = index / size_t(32);
//	unsigned int bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
//	return ((voxel_table[int_location]) & (1 << bit_pos));
//}

// Set a bit in the giant voxel table. This involves doing an atomic operation on a 32-bit word in memory.
// Blocking other threads writing to it for a very short time
__device__ __inline__ void setBit(unsigned int* voxel_table, size_t index){
	size_t int_location = index / size_t(32);
	unsigned int bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
	unsigned int mask = 1 << bit_pos;
	atomicOr(&(voxel_table[int_location]), mask);
}

// Main triangle voxelization method
__global__ void voxelize_triangle(voxinfo info, float* triangle_data, unsigned int* voxel_table, bool morton_order){
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	// Common variables used in the voxelization process
	float3 delta_p = make_float3(info.unit.x, info.unit.y, info.unit.z);
	int3 grid_max = make_int3(info.gridsize.x - 1, info.gridsize.y - 1, info.gridsize.z - 1); // grid max (grid runs from 0 to gridsize-1)

	while (thread_id < info.n_triangles){ // every thread works on specific triangles in its stride
		size_t t = thread_id * 9; // triangle contains 9 vertices

		// COMPUTE COMMON TRIANGLE PROPERTIES
		// Move vertices to origin using bbox
		float3 v0 = make_float3(triangle_data[t], triangle_data[t + 1], triangle_data[t + 2]) - info.bbox.min;
		float3 v1 = make_float3(triangle_data[t + 3], triangle_data[t + 4], triangle_data[t + 5]) - info.bbox.min;
		float3 v2 = make_float3(triangle_data[t + 6], triangle_data[t + 7], triangle_data[t + 8]) - info.bbox.min;
		// Edge vectors
		float3 e0 = v1 - v0;
		float3 e1 = v2 - v1;
		float3 e2 = v0 - v2;
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
		float3 c = make_float3(0.0f, 0.0f, 0.0f);
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
		for (int z = t_bbox_grid.min.z; z <= t_bbox_grid.max.z; z++){
			for (int y = t_bbox_grid.min.y; y <= t_bbox_grid.max.y; y++){
				for (int x = t_bbox_grid.min.x; x <= t_bbox_grid.max.x; x++){
					// if (checkBit(voxel_table, location)){ continue; }
#ifdef _DEBUG
					atomicAdd(&debug_d_n_voxels_tested, 1);
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
					atomicAdd(&debug_d_n_voxels_marked, 1);
#endif

					if (morton_order){
						size_t location = mortonEncode_LUT(x, y, z);
						setBit(voxel_table, location);
					} else {
						size_t location = 
							static_cast<size_t>(x) + 
							(static_cast<size_t>(y)* static_cast<size_t>(info.gridsize.x)) + 
							(static_cast<size_t>(z)* (static_cast<size_t>(info.gridsize.y)* static_cast<size_t>(info.gridsize.x)));
						setBit(voxel_table, location);
					}
					continue;
				}
			}
		}
#ifdef _DEBUG
		atomicAdd(&debug_d_n_triangles, 1);
#endif
		thread_id += stride;
	}
}

void voxelize(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool morton_code) {
	float elapsedTime;
	
	// Create timers, set start time
	cudaEvent_t start_vox, stop_vox;
	checkCudaErrors(cudaEventCreate(&start_vox));
	checkCudaErrors(cudaEventCreate(&stop_vox));

	// Copy morton LUT if we're encoding to morton
	if (morton_code){
		checkCudaErrors(cudaMemcpyToSymbol(morton256_x, host_morton256_x, 256 * sizeof(uint32_t)));
		checkCudaErrors(cudaMemcpyToSymbol(morton256_y, host_morton256_y, 256 * sizeof(uint32_t)));
		checkCudaErrors(cudaMemcpyToSymbol(morton256_z, host_morton256_z, 256 * sizeof(uint32_t)));
	}

	// Estimate best block and grid size using CUDA Occupancy Calculator
	int blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the  maximum occupancy for a full device launch 
	int gridSize;    // The actual grid size needed, based on input size 
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxelize_triangle, 0, 0);
	// Round up according to array size 
	gridSize = static_cast<int>((v.n_triangles + blockSize - 1) / blockSize);

	checkCudaErrors(cudaEventRecord(start_vox, 0));
	voxelize_triangle << <gridSize, blockSize >> > (v, triangle_data, vtable, morton_code);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaEventRecord(stop_vox, 0));
	checkCudaErrors(cudaEventSynchronize(stop_vox));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start_vox, stop_vox));
	printf("[Perf] Voxelization GPU time: %.1f ms\n", elapsedTime);

	// SANITY CHECKS
#ifdef _DEBUG
	size_t debug_n_triangles, debug_n_voxels_marked, debug_n_voxels_tested;
	checkCudaErrors(cudaMemcpyFromSymbol((void*)&(debug_n_triangles),debug_d_n_triangles, sizeof(debug_d_n_triangles), 0, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyFromSymbol((void*)&(debug_n_voxels_marked), debug_d_n_voxels_marked, sizeof(debug_d_n_voxels_marked), 0, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyFromSymbol((void*) & (debug_n_voxels_tested), debug_d_n_voxels_tested, sizeof(debug_d_n_voxels_tested), 0, cudaMemcpyDeviceToHost));
	printf("[Debug] Processed %llu triangles on the GPU \n", debug_n_triangles);
	printf("[Debug] Tested %llu voxels for overlap on GPU \n", debug_n_voxels_tested);
	printf("[Debug] Marked %llu voxels as filled (includes duplicates!) \n", debug_n_voxels_marked);
#endif

	// Destroy timers
	checkCudaErrors(cudaEventDestroy(start_vox));
	checkCudaErrors(cudaEventDestroy(stop_vox));
}
