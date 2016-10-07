#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "util_cuda.h"
#include "util_common.h"

__global__ void voxelize(size_t n_triangles, float* triangle_data){
	// determine max and min vector : this is a reduction problem
}
