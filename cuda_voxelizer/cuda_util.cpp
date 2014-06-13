#include "cuda_util.h"

// Check if CUDA requirements are met
int checkCudaRequirements(){
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
	fprintf(stdout,"Available global device memory: %llu bytes \n", properties.totalGlobalMem);
	return 1;
}