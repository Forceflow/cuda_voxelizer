#include "cuda_util.h"
#include "TriMesh.h"
#include <string>
#include <stdio.h>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include "util.h"

void voxelize(voxinfo v, float* triangle_data);

using namespace std;
string filename = "";
unsigned int gridsize = 1024;
float* triangles;

glm::vec3 trimesh_to_glm(trimesh::vec3 a){
	return glm::vec3(a[0], a[1], a[2]);
}

void parseProgramParameters(int argc, char* argv[]){
	if(argc<2){ // not enough arguments
		exit(0);
	} 
	for (int i = 1; i < argc; i++) {
		if (string(argv[i]) == "-f") {
			filename = argv[i + 1]; 
			i++;
		} else if (string(argv[i]) == "-s") {
			gridsize = atoi(argv[i + 1]);
			i++;
		}
	}
	fprintf(stdout, "Filename: %s \n", filename.c_str());
	fprintf(stdout, "Grid size: %i \n", gridsize);
}


int main(int argc, char *argv[]) {
	parseProgramParameters(argc, argv);
	checkCudaRequirements();

	trimesh::TriMesh *themesh = trimesh::TriMesh::read(filename.c_str());
	themesh->need_faces(); // unpack (possible) triangle strips so we have faces
	themesh->need_bbox(); // compute the bounding box

	fprintf(stdout, "Number of faces: %llu, faces table takes %llu kb \n", themesh->faces.size(), (size_t) (themesh->faces.size()*sizeof(trimesh::TriMesh::Face) / 1024.0f));
	fprintf(stdout, "Number of vertices: %llu, vertices table takes %llu kb \n", themesh->vertices.size(), (size_t) (themesh->vertices.size()*sizeof(trimesh::point) / 1024.0f));
	fprintf(stdout, "Flat triangle table will take %llu kb \n", (size_t) (themesh->faces.size()*9*sizeof(float) / 1024.0f));

	trianglesToMemory(themesh, &triangles);

	glm::vec3 bbox_min = trimesh_to_glm(themesh->bbox.min);
	glm::vec3 bbox_max = trimesh_to_glm(themesh->bbox.max);


	voxinfo v;

	v.unitlength = 1.0f;
	v.bbox_min = trimesh_to_glm(themesh->bbox.min);
	v.bbox_max = trimesh_to_glm(themesh->bbox.max);
	v.n_triangles = themesh->faces.size();
	v.gridsize = gridsize;

	//glm::vec3 test = glm::vec3(1,-2,4);
	//fprintf(stdout, " Before : %s \n", glm::to_string(test).c_str()); 
	//test = -test;
	//fprintf(stdout, " After : %s \n", glm::to_string(test).c_str());

	voxelize(v,triangles);
	

}