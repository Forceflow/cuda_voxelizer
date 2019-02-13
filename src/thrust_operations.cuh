// Trimesh for model importing
#include "TriMesh.h"
// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
// Thrust
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#define GLM_FORCE_CUDA
// GLM for maths
#include <glm/glm.hpp>
#include "util.h"

void trianglesToGPU_thrust(const trimesh::TriMesh *mesh, float** triangles);