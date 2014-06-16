#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "cuda_util.h"
#include "util.h"

// CUDA Global Memory variables
//__device__ size_t voxel_count = 0; // How many voxels did we count
//__device__ size_t triangles_seen_count = 0; // Sanity check

// Possible optimization: buffer bitsets (for now: too much overhead)
struct bufferedBitSetter{
	unsigned int* voxel_table;
	size_t current_int_location;
	unsigned int current_mask;

	__device__ __inline__ bufferedBitSetter(unsigned int* voxel_table, size_t index) :
		voxel_table(voxel_table), current_mask(0) {
		current_int_location = int(index / 32.0f);
	}

	__device__ __inline__ void setBit(size_t index){
		size_t new_int_location = int(index / 32.0f);
		if (current_int_location != new_int_location){
			flush();
			current_int_location = new_int_location;
		}
		unsigned int bit_pos = 31 - unsigned int(int(index) % 32);
		current_mask = current_mask | (1 << bit_pos);
	}

	__device__ __inline__ void flush(){
		if (current_mask != 0){
			atomicOr(&(voxel_table[current_int_location]), current_mask);
		}
	}
};

__device__ __inline__ bool checkBit(unsigned int* voxel_table, size_t index){
	size_t int_location = int(index / 32.0f);
	unsigned int bit_pos = 31 - unsigned int(int(index) % 32); // we count bit positions RtL, but array indices LtR
	return ((voxel_table[int_location]) & (1 << bit_pos));
}

__device__ __inline__ void setBit(unsigned int* voxel_table, size_t index){
	size_t int_location = int(index / 32.0f);
	unsigned int bit_pos = 31 - unsigned int(int(index) % 32); // we count bit positions RtL, but array indices LtR
	unsigned int mask = 1 << bit_pos;
	atomicOr(&(voxel_table[int_location]), mask);
}

// Main triangle voxelization method
__global__ void voxelize_triangle(voxinfo info, float* triangle_data, unsigned int* voxel_table){
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	// Common variables
	glm::vec3 delta_p = glm::vec3(info.unit, info.unit, info.unit);
	glm::vec3 c(0.0f, 0.0f, 0.0f); // critical point

	while (thread_id < info.n_triangles){ // every thread works on specific triangles in its stride
		size_t t = thread_id * 9; // triangle contains 9 vertices

		// COMPUTE COMMON TRIANGLE PROPERTIES
		glm::vec3 v0 = glm::vec3(triangle_data[t], triangle_data[t + 1], triangle_data[t + 2]) - info.bbox.min; // get v0 and move to origin
		glm::vec3 v1 = glm::vec3(triangle_data[t + 3], triangle_data[t + 4], triangle_data[t + 5]) - info.bbox.min; // get v1 and move to origin
		glm::vec3 v2 = glm::vec3(triangle_data[t + 6], triangle_data[t + 7], triangle_data[t + 8]) - info.bbox.min; // get v2 and move to origin
		glm::vec3 e0 = v1 - v0;
		glm::vec3 e1 = v2 - v1;
		glm::vec3 e2 = v0 - v2;
		glm::vec3 n = glm::normalize(glm::cross(e0, e1));

		//COMPUTE TRIANGLE BBOX IN GRID
		AABox<glm::vec3> t_bbox_world(glm::min(v0, glm::min(v1, v2)), glm::max(v0, glm::max(v1, v2)));
		AABox<glm::ivec3> t_bbox_grid;
		t_bbox_grid.min = glm::clamp(t_bbox_world.min / info.unit, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(info.gridsize, info.gridsize, info.gridsize));
		t_bbox_grid.max = glm::clamp(t_bbox_world.max / info.unit, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(info.gridsize, info.gridsize, info.gridsize));

		// PREPARE PLANE TEST PROPERTIES
		if (n.x > 0.0f) { c.x = info.unit; }
		if (n.y > 0.0f) { c.y = info.unit; }
		if (n.z > 0.0f) { c.z = info.unit; }
		float d1 = glm::dot(n, (c - v0));
		float d2 = glm::dot(n, ((delta_p - c) - v0));

		// PREPARE PROJECTION TEST PROPERTIES
		// XY plane
		glm::vec2 n_xy_e0(-1.0f*e0.y, e0.x);
		glm::vec2 n_xy_e1(-1.0f*e1.y, e1.x);
		glm::vec2 n_xy_e2(-1.0f*e2.y, e2.x);
		if (n.z < 0.0f) {
			n_xy_e0 = -n_xy_e0;
			n_xy_e1 = -n_xy_e1;
			n_xy_e2 = -n_xy_e2;
		}
		float d_xy_e0 = (-1.0f * glm::dot(n_xy_e0, glm::vec2(v0.x, v0.y))) + glm::max(0.0f, info.unit*n_xy_e0[0]) + glm::max(0.0f, info.unit*n_xy_e0[1]);
		float d_xy_e1 = (-1.0f * glm::dot(n_xy_e1, glm::vec2(v1.x, v1.y))) + glm::max(0.0f, info.unit*n_xy_e1[0]) + glm::max(0.0f, info.unit*n_xy_e1[1]);
		float d_xy_e2 = (-1.0f * glm::dot(n_xy_e2, glm::vec2(v2.x, v2.y))) + glm::max(0.0f, info.unit*n_xy_e2[0]) + glm::max(0.0f, info.unit*n_xy_e2[1]);
		// YZ plane
		glm::vec2 n_yz_e0(-1.0f*e0.z, e0.y);
		glm::vec2 n_yz_e1(-1.0f*e1.z, e1.y);
		glm::vec2 n_yz_e2(-1.0f*e2.z, e2.y);
		if (n.x < 0.0f) {
			n_yz_e0 = -n_yz_e0;
			n_yz_e1 = -n_yz_e1;
			n_yz_e2 = -n_yz_e2;
		}
		float d_yz_e0 = (-1.0f * glm::dot(n_yz_e0, glm::vec2(v0.y, v0.z))) + glm::max(0.0f, info.unit*n_yz_e0[0]) + glm::max(0.0f, info.unit*n_yz_e0[1]);
		float d_yz_e1 = (-1.0f * glm::dot(n_yz_e1, glm::vec2(v1.y, v1.z))) + glm::max(0.0f, info.unit*n_yz_e1[0]) + glm::max(0.0f, info.unit*n_yz_e1[1]);
		float d_yz_e2 = (-1.0f * glm::dot(n_yz_e2, glm::vec2(v2.y, v2.z))) + glm::max(0.0f, info.unit*n_yz_e2[0]) + glm::max(0.0f, info.unit*n_yz_e2[1]);
		// ZX plane
		glm::vec2 n_zx_e0(-1.0f*e0.x, e0.z);
		glm::vec2 n_zx_e1(-1.0f*e1.x, e1.z);
		glm::vec2 n_zx_e2(-1.0f*e2.x, e2.z);
		if (n.y < 0.0f) {
			n_zx_e0 = -n_zx_e0;
			n_zx_e1 = -n_zx_e1;
			n_zx_e2 = -n_zx_e2;
		}
		float d_xz_e0 = (-1.0f * glm::dot(n_zx_e0, glm::vec2(v0.z, v0.x))) + glm::max(0.0f, info.unit*n_zx_e0[0]) + glm::max(0.0f, info.unit*n_zx_e0[1]);
		float d_xz_e1 = (-1.0f * glm::dot(n_zx_e1, glm::vec2(v1.z, v1.x))) + glm::max(0.0f, info.unit*n_zx_e1[0]) + glm::max(0.0f, info.unit*n_zx_e1[1]);
		float d_xz_e2 = (-1.0f * glm::dot(n_zx_e2, glm::vec2(v2.z, v2.x))) + glm::max(0.0f, info.unit*n_zx_e2[0]) + glm::max(0.0f, info.unit*n_zx_e2[1]);

		// test possible grid boxes for overlap
		for (int z = t_bbox_grid.min.z; z <= t_bbox_grid.max.z; z++){
			for (int y = t_bbox_grid.min.y; y <= t_bbox_grid.max.y; y++){
				for (int x = t_bbox_grid.min.x; x <= t_bbox_grid.max.x; x++){
					size_t location = x + (y*info.gridsize) + (z*info.gridsize*info.gridsize);
					//if (checkBit(voxel_table, location)){ continue; }

					// TRIANGLE PLANE THROUGH BOX TEST
					glm::vec3 p(x*info.unit, y*info.unit, z*info.unit);
					float nDOTp = glm::dot(n, p);
					if ((nDOTp + d1) * (nDOTp + d2) > 0.0f){ continue; }

					// PROJECTION TESTS
					// XY
					glm::vec2 p_xy(p.x, p.y);
					if ((glm::dot(n_xy_e0, p_xy) + d_xy_e0) < 0.0f){ continue; }
					if ((glm::dot(n_xy_e1, p_xy) + d_xy_e1) < 0.0f){ continue; }
					if ((glm::dot(n_xy_e2, p_xy) + d_xy_e2) < 0.0f){ continue; }

					// YZ
					glm::vec2 p_yz(p.y, p.z);
					if ((glm::dot(n_yz_e0, p_yz) + d_yz_e0) < 0.0f){ continue; }
					if ((glm::dot(n_yz_e1, p_yz) + d_yz_e1) < 0.0f){ continue; }
					if ((glm::dot(n_yz_e2, p_yz) + d_yz_e2) < 0.0f){ continue; }

					// XZ	
					glm::vec2 p_zx(p.z, p.x);
					if ((glm::dot(n_zx_e0, p_zx) + d_xz_e0) < 0.0f){ continue; }
					if ((glm::dot(n_zx_e1, p_zx) + d_xz_e1) < 0.0f){ continue; }
					if ((glm::dot(n_zx_e2, p_zx) + d_xz_e2) < 0.0f){ continue; }

					setBit(voxel_table, location);
					continue;
				}
			}
		}

		// sanity check: atomically count triangles
		// atomicAdd(&triangles_seen_count, 1);
		thread_id += stride;
	}

}

void voxelize(voxinfo v, float* triangle_data, unsigned int* vtable){
	float* dev_triangle_data; // DEVICE pointer to triangle data
	unsigned int* dev_vtable; // DEVICE pointer to voxel_data
	float   elapsedTime;

	// Create timers, set start time
	cudaEvent_t start_total, stop_total, start_vox, stop_vox;
	HANDLE_CUDA_ERROR(cudaEventCreate(&start_total));
	HANDLE_CUDA_ERROR(cudaEventCreate(&stop_total));
	HANDLE_CUDA_ERROR(cudaEventCreate(&start_vox));
	HANDLE_CUDA_ERROR(cudaEventCreate(&stop_vox));
	HANDLE_CUDA_ERROR(cudaEventRecord(start_total, 0));

	// Malloc triangle memory
	HANDLE_CUDA_ERROR(cudaMalloc(&dev_triangle_data, v.n_triangles * 9 * sizeof(float)));
	HANDLE_CUDA_ERROR(cudaMemcpy(dev_triangle_data, (void*)triangle_data, v.n_triangles * 9 * sizeof(float), cudaMemcpyDefault));

	// Malloc voxelisation table
	size_t vtable_size = ((size_t)v.gridsize * v.gridsize * v.gridsize) / 8.0f;
	HANDLE_CUDA_ERROR(cudaMalloc(&dev_vtable, vtable_size));
	HANDLE_CUDA_ERROR(cudaMemset(dev_vtable, 0, vtable_size));

	HANDLE_CUDA_ERROR(cudaEventRecord(start_vox, 0));
	// if we pass triangle_data here directly, UVA takes care of memory transfer via DMA. Disabling for now.
	voxelize_triangle << <256, 256 >> >(v, dev_triangle_data, dev_vtable);
	CHECK_CUDA_ERROR();
	cudaDeviceSynchronize();
	HANDLE_CUDA_ERROR(cudaEventRecord(stop_vox, 0));
	HANDLE_CUDA_ERROR(cudaEventSynchronize(stop_vox));
	HANDLE_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start_vox, stop_vox));
	printf("Voxelisation GPU time:  %3.1f ms\n", elapsedTime);

	// SANITY CHECKS
	//size_t t_seen, v_count;
	//HANDLE_CUDA_ERROR(cudaMemcpyFromSymbol((void*)&(t_seen),triangles_seen_count, sizeof(t_seen), 0, cudaMemcpyDeviceToHost));
	//HANDLE_CUDA_ERROR(cudaMemcpyFromSymbol((void*)&(v_count), voxel_count, sizeof(v_count), 0, cudaMemcpyDeviceToHost));
	//printf("We've seen %llu triangles on the GPU \n", t_seen);
	//printf("We've found %llu voxels on the GPU \n", v_count);

	// Copy voxelisation table back to host
	HANDLE_CUDA_ERROR(cudaMemcpy(dev_vtable, (void*)vtable, vtable_size, cudaMemcpyDefault));

	// get stop time, and display the timing results
	HANDLE_CUDA_ERROR(cudaEventRecord(stop_total, 0));
	HANDLE_CUDA_ERROR(cudaEventSynchronize(stop_total));
	HANDLE_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start_total, stop_total));
	printf("Total GPU time (including memory transfers):  %3.1f ms\n", elapsedTime);

	// Destroy timers
	HANDLE_CUDA_ERROR(cudaEventDestroy(start_total));
	HANDLE_CUDA_ERROR(cudaEventDestroy(stop_total));
	HANDLE_CUDA_ERROR(cudaEventDestroy(start_vox));
	HANDLE_CUDA_ERROR(cudaEventDestroy(stop_vox));

	// Free memory
	HANDLE_CUDA_ERROR(cudaFree(dev_triangle_data));
	HANDLE_CUDA_ERROR(cudaFree(dev_vtable));
}
