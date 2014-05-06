#include "cuda_util.h"
#include "TriMesh.h"
#include <string>
#include <stdio.h>

extern void voxelize();

using namespace std;
string filename = "";
int gridsize = 1024;
float* triangles;

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

	voxelize();
	

}