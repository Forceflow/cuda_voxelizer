#ifndef VOXELIZE_H_
#define VOXELIZE_H_

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_check.cuh"

inline void allocateHostMem(size_t size, void **data){
	HANDLE_CUDA_ERROR(cudaHostAlloc(data, size, cudaHostAllocDefault));
}

// Check if CUDA requirements are met
inline int checkCudaRequirements(){
	// Is there a cuda device?
	int device_count = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&device_count));
	if(device_count < 1){
		fprintf(stderr, "No cuda devices found - we need at least one \n");
		return 0;
	} else {
		fprintf(stdout, "Found %i cuda devices, yay! \n", device_count);
	}

	// We'll be using first device by default
	cudaDeviceProp properties;
	HANDLE_CUDA_ERROR(cudaSetDevice(0));
	HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&properties,0));
	fprintf(stdout,"Device %d: \"%s\"\n", 0, properties.name);
	fprintf(stdout,"Available global device memory: %llu bytes", properties.totalGlobalMem);
	return 1;
}

#endif