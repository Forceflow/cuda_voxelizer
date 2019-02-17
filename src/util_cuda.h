#pragma once

// Standard libs
#include <stdio.h>
#include <cstdlib>
// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "libs/helper_cuda.h"
// GLM
#define GLM_FORCE_CUDA
#define GLM_FORCE_PURE
#include <glm/glm.hpp>


// Function to check cuda requirements
int initCuda();