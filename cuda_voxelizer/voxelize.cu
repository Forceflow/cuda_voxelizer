#include "voxelize.cuh"

// symbols for constant memory: the same for every triangle
__constant__ float3 model_bbox_min;
__constant__ float3 model_bbox_max;
__constant__ float unitlength;

__global__ void kernel()
{
	
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t voxelize()
{
    cudaError_t cudaStatus = cudaSuccess;
    return cudaStatus;
	
}
