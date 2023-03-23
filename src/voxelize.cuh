#pragma once

// Commun functions for both the solid and non-solid voxelization methods

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "util.h"
#include "util_cuda.h"
#include "libs/cuda/helper_math.h"
#include "morton_LUTs.h"

// Morton LUTs for when we need them
__constant__ uint32_t morton256_x[256];
__constant__ uint32_t morton256_y[256];
__constant__ uint32_t morton256_z[256];

// Encode morton code using LUT table
__device__ inline uint64_t mortonEncode_LUT(unsigned int x, unsigned int y, unsigned int z){
	uint64_t answer = 0;
	answer = morton256_z[(z >> 16) & 0xFF] |
		morton256_y[(y >> 16) & 0xFF] |
		morton256_x[(x >> 16) & 0xFF];
	answer = answer << 48 |
		morton256_z[(z >> 8) & 0xFF] |
		morton256_y[(y >> 8) & 0xFF] |
		morton256_x[(x >> 8) & 0xFF];
	answer = answer << 24 |
		morton256_z[(z)& 0xFF] |
		morton256_y[(y)& 0xFF] |
		morton256_x[(x)& 0xFF];
	return answer;
}