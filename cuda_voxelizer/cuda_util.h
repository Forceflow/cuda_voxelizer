#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_check.cuh"

void allocateHostMem(size_t size, void **data);
int checkCudaRequirements();

#endif