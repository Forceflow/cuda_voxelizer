#include <glm/glm.hpp>

// voxelisation info (same for every triangle)
struct voxinfo {
	size_t n_triangles;
	unsigned int gridsize;
	float unit;
	glm::vec3 bbox_min;
	glm::vec3 bbox_max;
};