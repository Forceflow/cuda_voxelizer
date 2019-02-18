#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLM_FORCE_CUDA
#define GLM_FORCE_PURE
#include <glm/glm.hpp>

#include <iostream>
#include "util.h"
#include "util_cuda.h"

#include "morton_LUTs.h"
