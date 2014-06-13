#ifndef UTIL_H_
#define UTIL_H_

#include <glm/glm.hpp>

template <typename T>
struct AABox {
	T min;
	T max;
	__device__ __host__ AABox() : min(T()), max(T()){}
	__device__ __host__ AABox(T min, T max) : min(min), max(max){}
};

// voxelisation info (same for every triangle)
struct voxinfo {
	AABox<glm::vec3> bbox;
	unsigned int gridsize;
	size_t n_triangles;
	float unit;

	voxinfo(AABox<glm::vec3> bbox, unsigned int gridsize, size_t n_triangles) 
		: gridsize(gridsize), bbox(bbox), n_triangles(n_triangles){
		unit = (bbox.max.x - bbox.min.x) / float (gridsize);
	}
};

template <typename T>
AABox<T> createMeshBBCube(AABox<T> box){
	AABox<T> answer(box.min, box.max);
	glm::vec3 lengths = box.max - box.min;
	float max_length = glm::max(lengths.x, glm::max(lengths.y, lengths.z));
	for (int i = 0; i<3; i++) {
		float delta = max_length - lengths[i];
		if (delta != 0){
			answer.min[i] = box.min[i] - (delta / 2.0f);
			answer.max[i] = box.max[i] + (delta / 2.0f);
		}
	}
	return answer;
}

#endif