#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "morton_LUTs.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <iostream>
#include "util.h"
#include "util_cuda.h"

void voxelize(const voxinfo & v, float* triangle_data, unsigned int* vtable, bool useMallocManaged, bool morton_code);