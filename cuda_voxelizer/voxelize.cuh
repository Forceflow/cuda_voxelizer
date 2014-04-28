#ifndef VOXELIZE_H_
#define VOXELIZE_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_check.cuh"
#include <stdio.h>

// Check if CUDA requirements are met
inline int checkCudaRequirements(){
	int device_count = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&device_count));
	if(device_count < 1){
		fprintf(stderr, "No cuda devices found - we need at least one");
		return 0;
	} else {
		fprintf(stdout, "Found %i cuda devices, yay!", device_count);
	}
	CHECK_CUDA_ERROR();
	return 1;
}

#endif