#pragma once

#include "Trimesh.h"
#include <glm/glm.hpp>

inline glm::vec3 trimesh_to_glm(trimesh::vec3 a){
	return glm::vec3(a[0], a[1], a[2]);
}

inline char checkVoxel(size_t x, size_t y, size_t z, size_t gridsize, const unsigned int* vtable){
	size_t location = x + (y*gridsize) + (z*gridsize*gridsize);
	size_t int_location = location / size_t(32);
	/*size_t max_index = (gridsize*gridsize*gridsize) / __int64(32);
	if (int_location >= max_index){
	fprintf(stdout, "Requested index too big: %llu \n", int_location);
	fprintf(stdout, "X %llu Y %llu Z %llu \n", int_location);
	}*/
	unsigned int bit_pos = size_t(31) - (location % size_t(32)); // we count bit positions RtL, but array indices LtR
	if ((vtable[int_location]) & (1 << bit_pos)){
		return char(1);
	}
	return char(0);
}