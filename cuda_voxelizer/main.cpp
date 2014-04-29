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

// Helper function to transfer triangles to CUDA-allocated pinned host memory
void trianglesToMemory(const trimesh::TriMesh *mesh, float** _data){
	// Allocate page-locked memory
	size_t size = sizeof(float)*9*(mesh->faces.size());
	fprintf(stdout,"Allocating %llu kb of page-locked host memory", (size_t) (size / 1024.0f));
	allocateHostMem(size, (void**) _data);
	// Loop over all triangles and place them in memory
	for(size_t i = 0; i < mesh->faces.size(); i++){
		const trimesh::point &v0 = mesh->vertices[mesh->faces[i][0]];
		const trimesh::point &v1 = mesh->vertices[mesh->faces[i][1]];
		const trimesh::point &v2 = mesh->vertices[mesh->faces[i][2]];
		size_t j = i*9;
		//memcpy((*_data)+j, &v0, 3*sizeof(float));
		//memcpy((*_data)+j+3, &v1, 3*sizeof(float));
		//memcpy((*_data)+j+6, &v2, 3*sizeof(float));
		(*_data)[j]   = v0[0];
		(*_data)[j+1] = v0[1];
		(*_data)[j+2] = v0[2];
		(*_data)[j+3] = v1[0];
		(*_data)[j+4] = v1[1];
		(*_data)[j+5] = v1[2];
		(*_data)[j+6] = v2[0];
		(*_data)[j+7] = v2[1];
		(*_data)[j+8] = v2[2];
	}
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