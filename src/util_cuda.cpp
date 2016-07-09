#include "util_cuda.h"

// Check if CUDA requirements are met
int checkCudaRequirements(){
	// Is there a cuda device?
	int device_count = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&device_count));
	if(device_count < 1){
		fprintf(stderr, "No cuda devices found - we need at least one. \n");
		return 0;
	} 

	// We'll be using first device by default
	cudaDeviceProp properties;
	HANDLE_CUDA_ERROR(cudaSetDevice(0));
	HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&properties,0));
	fprintf(stdout,"Device %d: \"%s\".\n", 0, properties.name);
	fprintf(stdout,"Available global device memory: %llu bytes. \n", properties.totalGlobalMem);

	// Check compute capability
	if (properties.major < 2){
		fprintf(stderr, "Your cuda device has compute capability %i.%i. We need at least 2.0 for atomic operations. \n", properties.major, properties.minor);
		return 0;
	} else {
		fprintf(stdout, "Compute capability: %i.%i.\n", properties.major, properties.minor);
	}


	return 1;
}