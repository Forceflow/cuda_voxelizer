#include "cuda_util.h"


// Helper function to transfer triangles to CUDA-allocated pinned host memory
void trianglesToMemory(const trimesh::TriMesh *mesh, float** _data){
	// Allocate page-locked memory
	size_t size = sizeof(float)*9*(mesh->faces.size());
	fprintf(stdout,"Allocating %llu kb of page-locked host memory", (size_t) (size / 1024.0f));
	HANDLE_CUDA_ERROR(cudaHostAlloc(_data, size, cudaHostAllocDefault));
	// Loop over all triangles and place them in memory
	for(size_t i = 0; i < mesh->faces.size(); i++){
		const trimesh::point &v0 = mesh->vertices[mesh->faces[i][0]];
		const trimesh::point &v1 = mesh->vertices[mesh->faces[i][1]];
		const trimesh::point &v2 = mesh->vertices[mesh->faces[i][2]];
		size_t j = i*9;
		//memcpy((*_data)+j, &v0, 3*sizeof(float));
		//memcpy((*_data)+j+3, &v1, 3*sizeof(float));
		//memcpy((*_data)+j+6, &v2, 3*sizeof(float));
		(*_data)[j]   = v0[0];
		(*_data)[j+1] = v0[1];
		(*_data)[j+2] = v0[2];
		(*_data)[j+3] = v1[0];
		(*_data)[j+4] = v1[1];
		(*_data)[j+5] = v1[2];
		(*_data)[j+6] = v2[0];
		(*_data)[j+7] = v2[1];
		(*_data)[j+8] = v2[2];
	}
}

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