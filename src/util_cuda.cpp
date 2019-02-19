#include "util_cuda.h"

// Check if CUDA requirements are met
int initCuda(){
	// Is there a CUDA device?
	int device_count = 0;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	if(device_count < 1){
		fprintf(stderr, "[CUDA] No CUDA devices found. \n \n");
		fprintf(stderr, "[CUDA] Make sure CUDA device is powered, connected and available. \n");
		fprintf(stderr, "[CUDA] On laptops: disable powersave/battery mode. \n");
		fprintf(stderr, "[CUDA] Exiting... \n");
		exit(0);
	}

	fprintf(stdout, "[CUDA] ");
	// Select best (fastest) CUDA device (method from Nvidia helper library)
	int device = findCudaDevice(0, 0);

	cudaDeviceProp properties;
	checkCudaErrors(cudaGetDeviceProperties(&properties,device));
	fprintf(stdout,"[CUDA] Available global device memory: %.0lf MB. \n", ((double) properties.totalGlobalMem / 1024 / 1024));

	// Check compute capability
	if (properties.major < 2){
		fprintf(stderr, "[CUDA] Your cuda device has compute capability %i.%i. We need at least 2.0 for atomic operations. Exiting. \n", properties.major, properties.minor);
		exit(0);
	}

	return 1;
}