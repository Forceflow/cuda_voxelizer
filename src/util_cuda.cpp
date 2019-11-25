#include "util_cuda.h"

// Check if CUDA requirements are met
bool initCuda(){

	int device_count = 0;
	// Check if CUDA runtime calls work at all
	cudaError t = cudaGetDeviceCount(&device_count);
	if (t != cudaSuccess) {
		fprintf(stderr, "[CUDA] First call to CUDA Runtime API failed. Are the drivers installed? \n");
		return false;
	}

	// Is there a CUDA device at all?
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	if(device_count < 1){
		fprintf(stderr, "[CUDA] No CUDA devices found. \n \n");
		fprintf(stderr, "[CUDA] Make sure CUDA device is powered, connected and available. \n");
		fprintf(stderr, "[CUDA] On laptops: disable powersave/battery mode. \n");
		fprintf(stderr, "[CUDA] Exiting... \n");
		return false;
	}

	fprintf(stderr, "[CUDA] CUDA device(s) found, picking best one \n");
	fprintf(stdout, "[CUDA] ");
	// We have at least 1 CUDA device, so now select the fastest (method from Nvidia helper library)
	int device = findCudaDevice(0, 0);

	// Print available device memory
	cudaDeviceProp properties;
	checkCudaErrors(cudaGetDeviceProperties(&properties,device));
	fprintf(stdout, "[CUDA] Best device: %s \n", properties.name);
	fprintf(stdout,"[CUDA] Available global device memory: %.0lf MB. \n", ((double) properties.totalGlobalMem / 1024 / 1024));

	// Check compute capability
	if (properties.major < 2){
		fprintf(stderr, "[CUDA] Your cuda device has compute capability %i.%i. We need at least 2.0 for atomic operations. Exiting. \n", properties.major, properties.minor);
		return false;
	}

	return true;
}