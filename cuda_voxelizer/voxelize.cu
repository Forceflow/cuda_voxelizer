#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_check.cuh"
#include "util.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// CUDA Global Memory
// Sanity check
__device__ size_t triangles_seen_count;

__global__ void voxelize_triangle(voxinfo info, float* triangle_data, bool* voxel_table){
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	//printf("Hi from thread %llu \n", thread_id);

	// Common variables
	glm::vec3 delta_p = glm::vec3(info.unit, info.unit, info.unit);
	glm::vec3 c(0.0f, 0.0f, 0.0f); // critical point

	while(thread_id < info.n_triangles){ // every thread works on specific triangles in its stride
		size_t t = thread_id*9; // triangle contains 9 vertices

		// COMPUTE COMMON TRIANGLE PROPERTIES
		glm::vec3 v0(triangle_data[t], triangle_data[t + 1], triangle_data[t + 2]);
		glm::vec3 v1(triangle_data[t + 3], triangle_data[t + 4], triangle_data[t + 5]);
		glm::vec3 v2(triangle_data[t + 6], triangle_data[t + 7], triangle_data[t + 8]);
		glm::vec3 e0 = v1 - v0;
		glm::vec3 e1 = v2 - v1;
		glm::vec3 e2 = v0 - v2;
		glm::vec3 n = glm::normalize(glm::cross(e0, e1));

		//COMPUTE TRIANGLE BBOX


		// PREPARE PLANE TEST PROPERTIES
		if (n.x > 0) { c.x = info.unit;}
		if (n.y > 0) { c.y = info.unit;}
		if (n.z > 0) { c.z = info.unit;}
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
		//for (int x = t_bbox_grid.min[0]; x <= t_bbox_grid.max[0]; x++){
		//	for (int y = t_bbox_grid.min[1]; y <= t_bbox_grid.max[1]; y++){
		//		for (int z = t_bbox_grid.min[2]; z <= t_bbox_grid.max[2]; z++){

//					uint64_t index = mortonEncode_LUT(z, y, x);
//
//					if (voxels[index - morton_start] == FULL_VOXEL){ continue; } // already marked, continue
//
//					// TRIANGLE PLANE THROUGH BOX TEST
//					vec3 p = vec3(x*unitlength, y*unitlength, z*unitlength);
//					float nDOTp = n DOT p;
//					if ((nDOTp + d1) * (nDOTp + d2) > 0.0f){ continue; }
//
//					// PROJECTION TESTS
//					// XY
//					vec2 p_xy = vec2(p[X], p[Y]);
//					if (((n_xy_e0 DOT p_xy) + d_xy_e0) < 0.0f){ continue; }
//					if (((n_xy_e1 DOT p_xy) + d_xy_e1) < 0.0f){ continue; }
//					if (((n_xy_e2 DOT p_xy) + d_xy_e2) < 0.0f){ continue; }
//
//					// YZ
//					vec2 p_yz = vec2(p[Y], p[Z]);
//					if (((n_yz_e0 DOT p_yz) + d_yz_e0) < 0.0f){ continue; }
//					if (((n_yz_e1 DOT p_yz) + d_yz_e1) < 0.0f){ continue; }
//					if (((n_yz_e2 DOT p_yz) + d_yz_e2) < 0.0f){ continue; }
//
//					// XZ	
//					vec2 p_zx = vec2(p[Z], p[X]);
//					if (((n_zx_e0 DOT p_zx) + d_xz_e0) < 0.0f){ continue; }
//					if (((n_zx_e1 DOT p_zx) + d_xz_e1) < 0.0f){ continue; }
//					if (((n_zx_e2 DOT p_zx) + d_xz_e2) < 0.0f){ continue; }
//
//#ifdef BINARY_VOXELIZATION
//					voxels[index - morton_start] = FULL_VOXEL;
//					if (use_data){ data.push_back(index); }
//#else
//					voxels[index - morton_start] = FULL_VOXEL;
//					data.push_back(VoxelData(index, t.normal, average3Vec(t.v0_color, t.v1_color, t.v2_color))); // we ignore data limits for colored voxelization
//#endif
//					nfilled++;
//					continue;
//				}
//			}
//		}

		// sanity check: atomically count triangles
		atomicAdd(&triangles_seen_count, 1);
		thread_id += stride;
	}
	
}

void voxelize(voxinfo v, float* triangle_data){
	float* dev_triangle_data; // DEVICE pointer to triangle data
	bool* dev_voxelisation_table; // DEVICE pointer to voxelisation table

    //cudaError_t cudaStatus = cudaSuccess;

	// Sanity check
	size_t t_seen = 0;
	HANDLE_CUDA_ERROR(cudaMemcpyToSymbol(triangles_seen_count, (void*) &(t_seen), sizeof(t_seen), 0, cudaMemcpyHostToDevice));

	// Malloc triangle memory
	HANDLE_CUDA_ERROR(cudaMalloc(&dev_triangle_data,v.n_triangles*9*sizeof(float)));
	HANDLE_CUDA_ERROR(cudaMemcpy(dev_triangle_data, (void*) triangle_data, v.n_triangles*9*sizeof(float), cudaMemcpyDefault));

	// allocate GPU memory for voxelization table
	//HANDLE_CUDA_ERROR(cudaMalloc

	// if we pass triangle_data here directly, UVA takes care of memory transfer via DMA. Disabling for now.
	voxelize_triangle<<<512,512>>>(v,dev_triangle_data,0);
	CHECK_CUDA_ERROR();

	cudaDeviceSynchronize();
	
	// Copy sanity check back to host
	HANDLE_CUDA_ERROR(cudaMemcpyFromSymbol((void*)&(t_seen),triangles_seen_count, sizeof(t_seen), 0, cudaMemcpyDeviceToHost));
	printf("We've seen %llu triangles on the GPU", t_seen);
	
    //return cudaStatus;
}
