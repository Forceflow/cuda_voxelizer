#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_check.cuh"
#include "TriMesh.h"
#include <glm/glm.hpp>
#include "util.h"

int checkCudaRequirements();

#endif