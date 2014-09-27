#include <string>
#include <stdio.h>
#include "binvox_io.h"
#include "util_cuda.h"
#include "TriMesh.h"
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include "util_common.h"

extern void voxelize(voxinfo v, float* triangle_data, unsigned int* vtable);

using namespace std;

// Default options
string filename = "";
string filename_base = "";
unsigned int gridsize = 64;

// Program data
float* triangles;
unsigned int* vtable;

// Limitations
size_t GPU_global_mem;

glm::vec3 trimesh_to_glm(trimesh::vec3 a){
	return glm::vec3(a[0], a[1], a[2]);
}

void printBits(size_t const size, void const * const ptr){
	unsigned char *b = (unsigned char*)ptr;
	unsigned char byte;
	int i, j;

	for (i = size - 1; i >= 0; i--){
		for (j = 7; j >= 0; j--){
			byte = b[i] & (1 << j);
			byte >>= j;
			if (byte){
				printf("X");
			}
			else {
				printf(".");
			}

			//printf("%u", byte);
		}
	}
	puts("");
}

// Helper function to transfer triangles to CUDA-allocated pinned host memory
void trianglesToMemory(const trimesh::TriMesh *mesh, float* triangles){
	// Loop over all triangles and place them in memory
	for (size_t i = 0; i < mesh->faces.size(); i++){
		const trimesh::point &v0 = mesh->vertices[mesh->faces[i][0]];
		const trimesh::point &v1 = mesh->vertices[mesh->faces[i][1]];
		const trimesh::point &v2 = mesh->vertices[mesh->faces[i][2]];
		size_t j = i * 9;
		memcpy((triangles) + j, &v0, 3 * sizeof(float));
		memcpy((triangles) + j + 3, &v1, 3 * sizeof(float));
		memcpy((triangles) + j + 6, &v2, 3 * sizeof(float));
	}
}

void parseProgramParameters(int argc, char* argv[]){
	if(argc<2){ // not enough arguments
		exit(0);
	} 
	for (int i = 1; i < argc; i++) {
		if (string(argv[i]) == "-f") {
			filename = argv[i + 1];
			filename_base = filename.substr(0, filename.find_last_of("."));
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
	fprintf(stdout, "\n## PROGRAM PARAMETERS \n");
	parseProgramParameters(argc, argv);
	fprintf(stdout, "\n## CUDA INIT \n");
	checkCudaRequirements();

	fprintf(stdout, "\n## MESH IMPORT \n");
	fflush(stdout);
	trimesh::TriMesh *themesh = trimesh::TriMesh::read(filename.c_str());
	themesh->need_faces(); // unpack (possible) triangle strips so we have faces
	themesh->need_bbox(); // compute the bounding box

	fprintf(stdout, "\n## MEMORY PREPARATION \n");
	fprintf(stdout, "Number of faces: %llu, faces table takes %llu kb \n", themesh->faces.size(), (size_t) (themesh->faces.size()*sizeof(trimesh::TriMesh::Face) / 1024.0f));
	fprintf(stdout, "Number of vertices: %llu, vertices table takes %llu kb \n", themesh->vertices.size(), (size_t) (themesh->vertices.size()*sizeof(trimesh::point) / 1024.0f));
	AABox<glm::vec3> bbox_mesh(trimesh_to_glm(themesh->bbox.min), trimesh_to_glm(themesh->bbox.max));

	size_t size = sizeof(float) * 9 * (themesh->faces.size());
	fprintf(stdout, "Allocating %llu kb of page-locked host memory \n", (size_t)(size / 1024.0f));
	HANDLE_CUDA_ERROR(cudaHostAlloc((void**) &triangles, size, cudaHostAllocDefault));
	fprintf(stdout, "Copy %llu triangles to page-locked host memory \n", (size_t)(themesh->faces.size()));
	trianglesToMemory(themesh, triangles);

	fprintf(stdout, "\n## VOXELISATION SETUP \n");
	voxinfo v(createMeshBBCube<glm::vec3>(bbox_mesh), gridsize, themesh->faces.size());
	fprintf(stdout, "Cubed mesh bbox = %s to %s \n", glm::to_string(v.bbox.min).c_str(), glm::to_string(v.bbox.max).c_str());
	fprintf(stdout, "Unit length = %f \n", v.unit);
	fprintf(stdout, "Grid size = %u \n", v.gridsize);
	size_t vtable_size = ((size_t)gridsize*gridsize*gridsize) / 8.0f;
	fprintf(stdout, "Need %llu kb for voxel table \n", size_t(vtable_size / 1024.0f));

	HANDLE_CUDA_ERROR(cudaHostAlloc((void **)&vtable, vtable_size, cudaHostAllocDefault));

	fprintf(stdout, "\n## GPU VOXELISATION \n");
	voxelize(v, triangles, vtable);

	// Sanity check
	//if (gridsize <= 64){
	//	for (size_t i = (vtable_size / 4.0f)-1; i > 0; i--){printBits(sizeof(int), &(vtable[i]));}
	//}

	string filename_output = filename_base + string("_") + to_string(gridsize) + string(".binvox");
	fprintf(stdout, "\n## OUTPUT TO BINVOX FILE \n");
	fprintf(stdout, "Filename: %s \n", filename_output.c_str());
	write_binvox(vtable, gridsize, filename_output);
}