#include "cuda_runtime.h"

// symbols for constant memory: the same for every triangle
__constant__ float3 model_bbox_min;
__constant__ float3 model_bbox_max;
__constant__ float unitlength;

__global__ void kernel()
{
	
}

// Helper function for using CUDA to add vectors in parallel.
void voxelize(float** triangles, )
{
    cudaError_t cudaStatus = cudaSuccess;
	kernel<<<1,1>>>();
    //return cudaStatus;

	
	
}
