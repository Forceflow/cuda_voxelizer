/*
	Solid voxelization based on the Schwarz-Seidel paper.
*/

#include "voxelize.cuh"

#ifdef _DEBUG
__device__ size_t debug_d_n_voxels_marked = 0;
__device__ size_t debug_d_n_triangles = 0;
__device__ size_t debug_d_n_voxels_tested = 0;
#endif

#define float_error 0.000001

// use Xor for voxels whose corresponding bits have to flipped
__device__ __inline__ void setBitXor(unsigned int* voxel_table, size_t index) {
	size_t int_location = index / size_t(32);
	unsigned int bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
	unsigned int mask = 1 << bit_pos;
	atomicXor(&(voxel_table[int_location]), mask);
}

//check the location with point and triangle
__device__ inline int check_point_triangle(glm::vec2 v0, glm::vec2 v1, glm::vec2 v2, glm::vec2 point)
{
	glm::vec2 PA = point - v0;
	glm::vec2 PB = point - v1;
	glm::vec2 PC = point - v2;

	float t1 = PA.x*PB.y - PA.y*PB.x;
	if (fabs(t1) < float_error&&PA.x*PB.x <= 0 && PA.y*PB.y <= 0)
		return 1;

	float t2 = PB.x*PC.y - PB.y*PC.x;
	if (fabs(t2) < float_error&&PB.x*PC.x <= 0 && PB.y*PC.y <= 0)
		return 2;

	float t3 = PC.x*PA.y - PC.y*PA.x;
	if (fabs(t3) < float_error&&PC.x*PA.x <= 0 && PC.y*PA.y <= 0)
		return 3;

	if (t1*t2 > 0 && t1*t3 > 0)
		return 0;
	else
		return -1;
}

//find the x coordinate of the voxel
__device__ inline float get_x_coordinate(glm::vec3 n, glm::vec3 v0, glm::vec2 point)
{
	return (-(n.y*(point.x - v0.y) + n.z*(point.y - v0.z)) / n.x + v0.x);
}

//check the triangle is counterclockwise or not
__device__ inline bool checkCCW(glm::vec2 v0, glm::vec2 v1, glm::vec2 v2)
{
	glm::vec2 e0 = v1 - v0;
	glm::vec2 e1 = v2 - v0;
	float result = e0.x*e1.y - e1.x*e0.y;
	if (result > 0)
		return true;
	else
		return false;
}

//top-left rule
__device__ inline bool TopLeftEdge(glm::vec2 v0, glm::vec2 v1)
{
	return ((v1.y<v0.y) || (v1.y == v0.y&&v0.x>v1.x));
}

//generate solid voxelization
__global__ void voxelize_triangle_solid(voxinfo info, float* triangle_data, unsigned int* voxel_table, bool morton_order)
{
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	while (thread_id < info.n_triangles) { // every thread works on specific triangles in its stride
		size_t t = thread_id * 9; // triangle contains 9 vertices

								  // COMPUTE COMMON TRIANGLE PROPERTIES
								  // Move vertices to origin using bbox
		glm::vec3 v0 = glm::vec3(triangle_data[t], triangle_data[t + 1], triangle_data[t + 2]) - info.bbox.min;
		glm::vec3 v1 = glm::vec3(triangle_data[t + 3], triangle_data[t + 4], triangle_data[t + 5]) - info.bbox.min;
		glm::vec3 v2 = glm::vec3(triangle_data[t + 6], triangle_data[t + 7], triangle_data[t + 8]) - info.bbox.min;
		// Edge vectors
		glm::vec3 e0 = v1 - v0;
		glm::vec3 e1 = v2 - v1;
		glm::vec3 e2 = v0 - v2;
		// Normal vector pointing up from the triangle
		glm::vec3 n = glm::normalize(glm::cross(e0, e1));
		if (fabs(n.x) < float_error)
			return;

		//Calculate the projection of three point into yoz plane
		glm::vec2 v0_yz = glm::vec2(v0.y, v0.z);
		glm::vec2 v1_yz = glm::vec2(v1.y, v1.z);
		glm::vec2 v2_yz = glm::vec2(v2.y, v2.z);

		//set the triangle counterclockwise
		if (!checkCCW(v0_yz, v1_yz, v2_yz))
		{
			glm::vec2 v3 = v1_yz;
			v1_yz = v2_yz;
			v2_yz = v3;
		}

		// COMPUTE TRIANGLE BBOX IN GRID
		// Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
		glm::vec2 bbox_max = glm::max(v0_yz, glm::max(v1_yz, v2_yz));
		glm::vec2 bbox_min = glm::min(v0_yz, glm::min(v1_yz, v2_yz));

		glm::vec2 bbox_max_grid = glm::vec2(floor(bbox_max.x / info.unit.y - 0.5), floor(bbox_max.y / info.unit.z - 0.5));
		glm::vec2 bbox_min_grid = glm::vec2(ceil(bbox_min.x / info.unit.y - 0.5), ceil(bbox_min.y / info.unit.z - 0.5));

		for (int y = bbox_min_grid.x; y <= bbox_max_grid.x; y++)
		{
			for (int z = bbox_min_grid.y; z <= bbox_max_grid.y; z++)
			{
				glm::vec2 point = glm::vec2((y + 0.5)*info.unit.y, (z + 0.5)*info.unit.z);
				int checknum = check_point_triangle(v0_yz, v1_yz, v2_yz, point);
				if ((checknum == 1 && TopLeftEdge(v0_yz, v1_yz)) || (checknum == 2 && TopLeftEdge(v1_yz, v2_yz)) || (checknum == 3 && TopLeftEdge(v2_yz, v0_yz)) || (checknum == 0))
				{
					int xmax = int(get_x_coordinate(n, v0, point) / info.unit.x - 0.5);
					for (int x = 0; x <= xmax; x++)
					{
						if (morton_order){
							size_t location = mortonEncode_LUT(x, y, z);
							setBitXor(voxel_table, location);
						} else {
							size_t location = static_cast<size_t>(x) + (static_cast<size_t>(y)* static_cast<size_t>(info.gridsize.y)) + (static_cast<size_t>(z)* static_cast<size_t>(info.gridsize.y)* static_cast<size_t>(info.gridsize.z));
							setBitXor(voxel_table, location);
						}
						continue;
					}
				}
			}
		}

		// sanity check: atomically count triangles
		//atomicAdd(&triangles_seen_count, 1);
		thread_id += stride;
	}
}

void voxelize_solid(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code) {
	float   elapsedTime;

	// These are only used when we're not using UNIFIED memory
	unsigned int* dev_vtable; // DEVICE pointer to voxel_data
	size_t vtable_size; // vtable size
	
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
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxelize_triangle_solid, 0, 0);
	// Round up according to array size 
	gridSize = (v.n_triangles + blockSize - 1) / blockSize;

	if (useThrustPath) { // We're not using UNIFIED memory
		vtable_size = ((size_t)v.gridsize.x * v.gridsize.y * v.gridsize.z) / (size_t) 8.0;
		fprintf(stdout, "[Voxel Grid] Allocating %llu kB of DEVICE memory for Voxel Grid\n", size_t(vtable_size / 1024.0f));
		checkCudaErrors(cudaMalloc(&dev_vtable, vtable_size));
		checkCudaErrors(cudaMemset(dev_vtable, 0, vtable_size));
		// Start voxelization
		checkCudaErrors(cudaEventRecord(start_vox, 0));
		voxelize_triangle_solid << <gridSize, blockSize >> > (v, triangle_data, dev_vtable, morton_code);
	}
	else { // UNIFIED MEMORY 
		checkCudaErrors(cudaEventRecord(start_vox, 0));
		voxelize_triangle_solid << <gridSize, blockSize >> > (v, triangle_data, vtable, morton_code);
	}

	cudaDeviceSynchronize();
	checkCudaErrors(cudaEventRecord(stop_vox, 0));
	checkCudaErrors(cudaEventSynchronize(stop_vox));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start_vox, stop_vox));
	printf("[Perf] Voxelization GPU time: %.1f ms\n", elapsedTime);

	// If we're not using UNIFIED memory, copy the voxel table back and free all
	if (useThrustPath){
		fprintf(stdout, "[Voxel Grid] Copying %llu kB to page-locked HOST memory\n", size_t(vtable_size / 1024.0f));
		checkCudaErrors(cudaMemcpy((void*)vtable, dev_vtable, vtable_size, cudaMemcpyDefault));
		fprintf(stdout, "[Voxel Grid] Freeing %llu kB of DEVICE memory\n", size_t(vtable_size / 1024.0f));
		checkCudaErrors(cudaFree(dev_vtable));
	}

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