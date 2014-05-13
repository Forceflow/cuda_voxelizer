#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_check.cuh"
#include <glm/glm.hpp>
#include "util.h"

__global__ void voxelize_triangle(voxinfo info, float* triangle_data, bool* voxel_table){
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	//printf("Thread %i saying hi \n", thread_id);
	
	using namespace glm; // we use GLM for all the vector operations

	// every thread works on specific triangles
	while(thread_id < info.n_triangles){
		//printf("Looking at triangle %i \n", thread_id);
		size_t t = thread_id*9;

		// compute common properties for triangle
		vec3 delta_p = vec3(info.unitlength, info.unitlength, info.unitlength);
		vec3 v0 = vec3(triangle_data[t], triangle_data[t+1], triangle_data[t+2]);
		vec3 v1 = vec3(triangle_data[t+3], triangle_data[t+4], triangle_data[t+5]);
		vec3 v2 = vec3(triangle_data[t+6], triangle_data[t+7], triangle_data[t+8]);
		vec3 e0 = v1-v0;
		vec3 e1 = v2-v1;
		vec3 e2 = v0-v2;
		vec3 n = normalize(cross(e0,e1));

		// PLANE TEST PROPERTIES
		vec3 c = vec3(0.0f, 0.0f, 0.0f); // critical point
		if (n.x > 0) { c.x = info.unitlength;}
		if (n.y > 0) { c.y = info.unitlength;}
		if (n.z > 0) { c.z = info.unitlength;}
		float d1 = dot(n, (c - v0));
		float d2 = dot(n, ((delta_p - c) - v0));

		// PROJECTION TEST PROPERTIES
		// XY plane
		vec2 n_xy_e0 = vec2(-1.0f*e0.y, e0.x);
		vec2 n_xy_e1 = vec2(-1.0f*e1.y, e1.x);
		vec2 n_xy_e2 = vec2(-1.0f*e2.y, e2.x);
		if (n.z < 0.0f) {
			n_xy_e0 = -n_xy_e0;
			n_xy_e1 = -n_xy_e1;
			n_xy_e2 = -n_xy_e2;
		}
		float d_xy_e0 = (-1.0f * dot(n_xy_e0, vec2(v0.x, v0.y))) + max(0.0f, info.unitlength*n_xy_e0.x) + max(0.0f, info.unitlength*n_xy_e0.y);
		float d_xy_e1 = (-1.0f * dot(n_xy_e1, vec2(v1.x, v1.y))) + max(0.0f, info.unitlength*n_xy_e1.x) + max(0.0f, info.unitlength*n_xy_e1.y);
		float d_xy_e2 = (-1.0f * dot(n_xy_e2, vec2(v2.x, v2.y))) + max(0.0f, info.unitlength*n_xy_e2.x) + max(0.0f, info.unitlength*n_xy_e2.y);
		// YZ plane
		vec2 n_yz_e0 = vec2(-1.0f*e0.z, e0.y);
		vec2 n_yz_e1 = vec2(-1.0f*e1.z, e1.y);
		vec2 n_yz_e2 = vec2(-1.0f*e2.z, e2.y);
		if (n.x < 0.0f) {
			n_yz_e0 = -1.0f * n_yz_e0;
			n_yz_e1 = -1.0f * n_yz_e1;
			n_yz_e2 = -1.0f * n_yz_e2;
		}
		//float d_yz_e0 = (-1.0f * (n_yz_e0 DOT vec2(t.v0[Y], t.v0[Z]))) + max(0.0f, unitlength*n_yz_e0[0]) + max(0.0f, unitlength*n_yz_e0[1]);
		//float d_yz_e1 = (-1.0f * (n_yz_e1 DOT vec2(t.v1[Y], t.v1[Z]))) + max(0.0f, unitlength*n_yz_e1[0]) + max(0.0f, unitlength*n_yz_e1[1]);
		//float d_yz_e2 = (-1.0f * (n_yz_e2 DOT vec2(t.v2[Y], t.v2[Z]))) + max(0.0f, unitlength*n_yz_e2[0]) + max(0.0f, unitlength*n_yz_e2[1]);
		//// ZX plane
		//vec2 n_zx_e0 = vec2(-1.0f*e0[X], e0[Z]);
		//vec2 n_zx_e1 = vec2(-1.0f*e1[X], e1[Z]);
		//vec2 n_zx_e2 = vec2(-1.0f*e2[X], e2[Z]);
		//if (n[Y] < 0.0f) {
		//	n_zx_e0 = -1.0f * n_zx_e0;
		//	n_zx_e1 = -1.0f * n_zx_e1;
		//	n_zx_e2 = -1.0f * n_zx_e2;
		//}
		//float d_xz_e0 = (-1.0f * (n_zx_e0 DOT vec2(t.v0[Z], t.v0[X]))) + max(0.0f, unitlength*n_zx_e0[0]) + max(0.0f, unitlength*n_zx_e0[1]);
		//float d_xz_e1 = (-1.0f * (n_zx_e1 DOT vec2(t.v1[Z], t.v1[X]))) + max(0.0f, unitlength*n_zx_e1[0]) + max(0.0f, unitlength*n_zx_e1[1]);
		//float d_xz_e2 = (-1.0f * (n_zx_e2 DOT vec2(t.v2[Z], t.v2[X]))) + max(0.0f, unitlength*n_zx_e2[0]) + max(0.0f, unitlength*n_zx_e2[1]);

		thread_id += blockDim.x * gridDim.x;
	}
	
}

void voxelize(voxinfo v, float* triangle_data){
    cudaError_t cudaStatus = cudaSuccess;
	cudaDeviceSynchronize();

	// copy triangle data to GPU
	float* dev_triangle_data;
	HANDLE_CUDA_ERROR(cudaMalloc(&dev_triangle_data,v.n_triangles*9*sizeof(float)));
	HANDLE_CUDA_ERROR(cudaMemcpy(dev_triangle_data, (void*) triangle_data, v.n_triangles*9*sizeof(float), cudaMemcpyDefault));

	// if we pass triangle_data here directly, UVA takes care of memory transfer via DMA. Disabling for now.
	voxelize_triangle<<<256,256>>>(v,dev_triangle_data,0);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR();

    //return cudaStatus;

	
	
}
