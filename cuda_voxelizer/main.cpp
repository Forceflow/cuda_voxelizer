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
inline void trianglesToMemory(const trimesh::TriMesh *mesh, float** _triangles){
	// allocate pinned host memory
	//HANDLE_CUDA_ERROR(cudaHostAlloc(_triangles,sizeof(float)*9*mesh->faces.size(),cudaHostAllocDefault));
	// loop over all triangles and place them in memory
	trimesh::point v0,v1,v2;
	for(size_t i = 0; i < mesh->faces.size(); i++){
		v0 = mesh->vertices[mesh->faces[i][0]];
		v1 = mesh->vertices[mesh->faces[i][1]];
		v2 = mesh->vertices[mesh->faces[i][2]];
	}
}


int main(int argc, char *argv[]) {
	parseProgramParameters(argc, argv);
	checkCudaRequirements();

	allocateHostMem(80, (void**) &triangles);

	voxelize();

	trimesh::TriMesh *themesh = trimesh::TriMesh::read(filename.c_str());
	themesh->need_faces(); // unpack (possible) triangle strips so we have faces
	themesh->need_bbox(); // compute the bounding box

	fprintf(stdout, "Number of faces: %ull, faces table takes %ull bytes \n", themesh->faces.size(), themesh->faces.size()*sizeof(trimesh::TriMesh::Face));
	fprintf(stdout, "Number of vertices: %ull, vertices table takes %ull bytes \n", themesh->vertices.size(), themesh->vertices.size()*sizeof(trimesh::point));
	fprintf(stdout, "Flat triangle table will take %ull bytes \n", themesh->faces.size()*9*sizeof(float));

	

}