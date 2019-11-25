#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN // Please, not too much windows shenanigans
#endif

// Standard libs
#include <string>
#include <cstdio>
// GLM for maths
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
// Trimesh for model importing
#include "TriMesh.h"
// Util
#include "util.h"
#include "util_io.h"
#include "util_cuda.h"
#include "timer.h"

using namespace std;
string version_number = "v0.4.1";

// Forward declaration of CUDA functions
float* meshToGPU_thrust(const trimesh::TriMesh *mesh); // METHOD 3 to transfer triangles can be found in thrust_operations.cu(h)
void cleanup_thrust();
void voxelize(const voxinfo & v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);

// Output formats
enum class OutputFormat { output_binvox = 0, output_morton = 1, output_obj = 2};
char *OutputFormats[] = { "binvox file", "morton encoded blob", "obj file"};

// Default options
string filename = "";
string filename_base = "";
OutputFormat outputformat = OutputFormat::output_binvox;
unsigned int gridsize = 256;
bool useThrustPath = false;

void printHeader(){
	cout << "CUDA Voxelizer " << version_number << " by Jeroen Baert" << endl; 
	cout << "github.com/Forceflow/cuda_voxelizer - jeroen.baert@cs.kuleuven.be" << endl;
}

void printExample() {
	cout << "Example: cuda_voxelizer -f /home/jeroen/bunny.ply -s 512" << endl;
}

void printHelp(){
	fprintf(stdout, "\n## HELP  \n");
	cout << "Program options: " << endl;
	cout << " -f <path to model file: .ply, .obj, .3ds> (required)" << endl;
	cout << " -s <voxelization grid size, power of 2: 8 -> 512, 1024, ... (default: 256)>" << endl;
	cout << " -o <output format: binvox, obj or morton (default: binvox)>" << endl;
	cout << " -t : Force using CUDA Thrust Library (possible speedup / throughput improvement)" << endl;
	printExample();
}

// METHOD 1: Helper function to transfer triangles to automatically managed CUDA memory ( > CUDA 7.x)
float* meshToGPU_managed(const trimesh::TriMesh *mesh) {
	Timer t; t.start();
	size_t n_floats = sizeof(float) * 9 * (mesh->faces.size());
	float* device_triangles;
	fprintf(stdout, "[Mesh] Allocating %s of CUDA-managed UNIFIED memory for triangle data \n", (readableSize(n_floats)).c_str());
	checkCudaErrors(cudaMallocManaged((void**) &device_triangles, n_floats)); // managed memory
	fprintf(stdout, "[Mesh] Copy %llu triangles to CUDA-managed UNIFIED memory \n", (size_t)(mesh->faces.size()));
	for (size_t i = 0; i < mesh->faces.size(); i++) {
		glm::vec3 v0 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
		glm::vec3 v1 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
		glm::vec3 v2 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
		size_t j = i * 9;
		memcpy((device_triangles)+j, glm::value_ptr(v0), sizeof(glm::vec3));
		memcpy((device_triangles)+j+3, glm::value_ptr(v1), sizeof(glm::vec3));
		memcpy((device_triangles)+j+6, glm::value_ptr(v2), sizeof(glm::vec3));
	}
	t.stop();fprintf(stdout, "[Mesh] Transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds);
	return device_triangles;
}

//// METHOD 2: Helper function to transfer triangles to old-style, self-managed CUDA memory ( < CUDA 7.x )
//float* meshToGPU(const trimesh::TriMesh *mesh){
//	size_t n_floats = sizeof(float) * 9 * (mesh->faces.size());
//	float* pagelocktriangles;
//	fprintf(stdout, "Allocating %llu kb of page-locked HOST memory \n", (size_t)(n_floats / 1024.0f));
//	checkCudaErrors(cudaHostAlloc((void**)&pagelocktriangles, n_floats, cudaHostAllocDefault)); // pinned memory to easily copy from
//	fprintf(stdout, "Copy %llu triangles to page-locked HOST memory \n", (size_t)(mesh->faces.size()));
//	for (size_t i = 0; i < mesh->faces.size(); i++){
//		glm::vec3 v0 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
//		glm::vec3 v1 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
//		glm::vec3 v2 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
//		size_t j = i * 9;
//		memcpy((pagelocktriangles)+j, glm::value_ptr(v0), sizeof(glm::vec3));
//		memcpy((pagelocktriangles)+j+3, glm::value_ptr(v1), sizeof(glm::vec3));
//		memcpy((pagelocktriangles)+j+6, glm::value_ptr(v2), sizeof(glm::vec3));
//	}
//	float* device_triangles;
//	fprintf(stdout, "Allocating %llu kb of DEVICE memory \n", (size_t)(n_floats / 1024.0f));
//	checkCudaErrors(cudaMalloc((void **) &device_triangles, n_floats));
//	fprintf(stdout, "Copy %llu triangles from page-locked HOST memory to DEVICE memory \n", (size_t)(mesh->faces.size()));
//	checkCudaErrors(cudaMemcpy((void *) device_triangles, (void*) pagelocktriangles, n_floats, cudaMemcpyDefault));
//	return device_triangles;
//}



// Parse the program parameters and set them as global variables
void parseProgramParameters(int argc, char* argv[]){
	if(argc<2){ // not enough arguments
		fprintf(stdout, "Not enough program parameters. \n \n");
		printHelp();
		exit(0);
	} 
	bool filegiven = false;
	for (int i = 1; i < argc; i++) {
		if (string(argv[i]) == "-f") {
			filename = argv[i + 1];
			filename_base = filename.substr(0, filename.find_last_of("."));
			filegiven = true;
			if (!file_exists(filename)) {
				fprintf(stdout, "[Err] File does not exist / cannot access: %s \n", filename.c_str());
				exit(1);
			}
			i++;
		}
		else if (string(argv[i]) == "-s") {
			gridsize = atoi(argv[i + 1]);
			i++;
		} else if (string(argv[i]) == "-h") {
			printHelp();
			exit(0);
		} else if (string(argv[i]) == "-o") {
			string output = (argv[i + 1]);
			transform(output.begin(), output.end(), output.begin(), ::tolower); // to lowercase
			if (output == "binvox"){
				outputformat = OutputFormat::output_binvox;
			}
			else if (output == "morton"){
				outputformat = OutputFormat::output_morton;
			}
			else if (output == "obj") {
				outputformat = OutputFormat::output_obj;
			}
			else {
				fprintf(stdout, "[Err] Unrecognized output format: %s, valid options are binvox (default) or morton \n", output.c_str());
				exit(1);
			}
		}
		else if (string(argv[i]) == "-t") {
			useThrustPath = true;
		}
	}
	if (!filegiven) {
		fprintf(stdout, "[Err] You didn't specify a file using -f (path). This is required. Exiting. \n");
		printExample();
		exit(1);
	}
	fprintf(stdout, "[Info] Filename: %s \n", filename.c_str());
	fprintf(stdout, "[Info] Grid size: %i \n", gridsize);
	fprintf(stdout, "[Info] Output format: %s \n", OutputFormats[int(outputformat)]);
	fprintf(stdout, "[Info] Using CUDA Thrust: %s \n", useThrustPath ? "Yes" : "No");
}

int main(int argc, char *argv[]) {
	Timer t; t.start();
	printHeader();
	fprintf(stdout, "\n## PROGRAM PARAMETERS \n");
	parseProgramParameters(argc, argv);
	fprintf(stdout, "\n## CUDA INIT \n");
	initCuda();

	fflush(stdout);
	trimesh::TriMesh::set_verbose(false);
#ifdef _DEBUG
	fprintf(stdout, "\n## MESH IMPORT \n");
	trimesh::TriMesh::set_verbose(true);
#endif
	fprintf(stdout, "\n## Read input mesh \n");
	fprintf(stdout, "[I/O] Reading mesh from %s \n", filename.c_str());
	trimesh::TriMesh *themesh = trimesh::TriMesh::read(filename.c_str());
	fprintf(stdout, "[Mesh] Computing faces \n");
	themesh->need_faces(); // Trimesh: Unpack (possible) triangle strips so we have faces for sure
	fprintf(stdout, "[Mesh] Computing bbox \n");
	themesh->need_bbox(); // Trimesh: Compute the bounding box (in model coordinates)

	fprintf(stdout, "\n## TRIANGLES TO GPU TRANSFER \n");
	fprintf(stdout, "[Mesh] Number of faces: %zu \n", themesh->faces.size());
	fprintf(stdout, "[Mesh] Number of vertices: %zu \n", themesh->vertices.size());
	float* device_triangles;

	if(useThrustPath){
		device_triangles = meshToGPU_thrust(themesh);
	}
	else {
		device_triangles = meshToGPU_managed(themesh);
	}

	fprintf(stdout, "\n## VOXELISATION SETUP \n");
	// Initialize our own AABox
	AABox<glm::vec3> bbox_mesh(trimesh_to_glm(themesh->bbox.min), trimesh_to_glm(themesh->bbox.max)); 
	// Transform that AABox to a cubical box (by padding directions if needed)
	voxinfo v(createMeshBBCube<glm::vec3>(bbox_mesh), glm::uvec3(gridsize, gridsize, gridsize), themesh->faces.size());
	v.print();
	size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(v.gridsize.x)* static_cast<size_t>(v.gridsize.y)* static_cast<size_t>(v.gridsize.z)) / 8.0f);

	unsigned int* vtable;
	if (!useThrustPath) {
		fprintf(stdout, "[Voxel Grid] Allocating %s of CUDA-managed UNIFIED memory for Voxel Grid\n", readableSize(vtable_size).c_str());
		checkCudaErrors(cudaMallocManaged((void **)&vtable, vtable_size));
	}
	else{
		// ALLOCATE MEMORY ON HOST
		fprintf(stdout, "[Voxel Grid] Allocating %s kB of page-locked HOST memory for Voxel Grid\n", readableSize(vtable_size).c_str());
		checkCudaErrors(cudaHostAlloc((void **)&vtable, vtable_size, cudaHostAllocDefault));
	}
	fprintf(stdout, "\n## GPU VOXELISATION \n");
	voxelize(v, device_triangles, vtable, useThrustPath, (outputformat == OutputFormat::output_morton));

	if (outputformat == OutputFormat::output_morton){
		fprintf(stdout, "\n## OUTPUT TO BINARY FILE \n");
		write_binary(vtable, vtable_size, filename);
	} else if (outputformat == OutputFormat::output_binvox){
		fprintf(stdout, "\n## OUTPUT TO BINVOX FILE \n");
		write_binvox(vtable, gridsize, filename);
	}
	else if (outputformat == OutputFormat::output_obj) {
		write_obj(vtable, gridsize, filename);
	}

	if (useThrustPath) {
		cleanup_thrust();
	}
	t.stop(); fprintf(stdout, "[Info] Total runtime: %.1f ms \n", t.elapsed_time_milliseconds);
}
