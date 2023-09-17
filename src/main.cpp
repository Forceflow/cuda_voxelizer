#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN // Please, not too much windows shenanigans
#endif

// Standard libs
#include <string>
#include <cstdio>

// Trimesh for model importing
#include "TriMesh.h"
// Util
#include "util.h"
#include "util_io.h"
#include "util_cuda.h"
#include "timer.h"
// CPU voxelizer fallback
#include "cpu_voxelizer.h"

using namespace std;
string version_number = "v0.6";

// Forward declaration of CUDA functions
void voxelize(const voxinfo & v, float* triangle_data, unsigned int* vtable, bool morton_code);
void voxelize_solid(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool morton_code);

// Output formats
enum class OutputFormat { output_binvox = 0, output_morton = 1, output_obj_points = 2, output_obj_cubes = 3, output_vox = 4};
char *OutputFormats[] = { "binvox file", "morton encoded blob", "obj file (pointcloud)", "obj file (cubes)", "magicavoxel file"};

// Default options
string filename = "";
string filename_base = "";
OutputFormat outputformat = OutputFormat::output_vox;
unsigned int gridsize = 256;
bool forceCPU = false;
bool solidVoxelization = false;

void printHeader(){
	fprintf(stdout, "## CUDA VOXELIZER \n");
	cout << "CUDA Voxelizer " << version_number << " by Jeroen Baert" << endl; 
	cout << "https://github.com/Forceflow/cuda_voxelizer - mail (at) jeroen-baert (dot) be" << endl;
}

void printExample() {
	cout << "Example: cuda_voxelizer -f /home/jeroen/bunny.ply -s 512" << endl;
}

void printHelp(){
	fprintf(stdout, "\n## HELP  \n");
	cout << "Program options: " << endl << endl;
	cout << " -f <path to model file: .ply, .obj, .3ds> (required)" << endl;
	cout << " -s <voxelization grid size, power of 2: 8 -> 512, 1024, ... (default: 256)>" << endl;
	cout << " -o <output format: vox, binvox, obj, obj_points or morton (default: vox)>" << endl;
	cout << " -cpu : Force CPU-based voxelization (slow, but works if no compatible GPU can be found)" << endl;
	cout << " -solid : Force solid voxelization (experimental, needs watertight model)" << endl << endl;
	printExample();
	cout << endl;
}

// METHOD 1: Helper function to transfer triangles to automatically managed CUDA memory ( > CUDA 7.x)
float* meshToGPU_managed(const trimesh::TriMesh *mesh) {
	Timer t; t.start();
	size_t n_floats = sizeof(float) * 9 * (mesh->faces.size());
	float* device_triangles = 0;
	fprintf(stdout, "[Mesh] Allocating %s of CUDA-managed UNIFIED memory for triangle data \n", (readableSize(n_floats)).c_str());
	checkCudaErrors(cudaMallocManaged((void**) &device_triangles, n_floats)); // managed memory
	fprintf(stdout, "[Mesh] Copy %llu triangles to CUDA-managed UNIFIED memory \n", (size_t)(mesh->faces.size()));
	for (size_t i = 0; i < mesh->faces.size(); i++) {
		float3 v0 = trimesh_to_float3<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
		float3 v1 = trimesh_to_float3<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
		float3 v2 = trimesh_to_float3<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
		size_t j = i * 9;
		// Memcpy assuming the floats are laid out next to eachother
		memcpy((device_triangles)+j, &v0.x, 3*sizeof(float)); 
		memcpy((device_triangles)+j+3, &v1.x, 3*sizeof(float));
		memcpy((device_triangles)+j+6, &v2.x, 3*sizeof(float));
	}
	t.stop();fprintf(stdout, "[Perf] Mesh transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds);
	return device_triangles;
}

// METHOD 2: Helper function to transfer triangles to old-style, self-managed CUDA memory ( < CUDA 7.x )
// Leaving this here for reference, the function above should be faster and better managed on all versions CUDA 7+
// 
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
			if (!file_exists(filename)) {fprintf(stdout, "[Err] File does not exist / cannot access: %s \n", filename.c_str());exit(1);}
			i++;
		}
		else if (string(argv[i]) == "-s") {
			gridsize = atoi(argv[i + 1]);
			i++;
		} else if (string(argv[i]) == "-h") {
			printHelp(); exit(0);
		} else if (string(argv[i]) == "-o") {
			string output = (argv[i + 1]);
			transform(output.begin(), output.end(), output.begin(), ::tolower); // to lowercase
			if (output == "binvox"){outputformat = OutputFormat::output_binvox;}
			else if (output == "morton"){outputformat = OutputFormat::output_morton;}
			else if (output == "obj"){outputformat = OutputFormat::output_obj_cubes;}
			else if (output == "obj_points") { outputformat = OutputFormat::output_obj_points; }
			else if (output == "vox") { outputformat = OutputFormat::output_vox; }
			else {fprintf(stdout, "[Err] Unrecognized output format: %s, valid options are binvox (default), morton, obj or obj_points \n", output.c_str());exit(1);}
		}
		else if (string(argv[i]) == "-cpu") {
			forceCPU = true;
		}
		else if (string(argv[i])=="-solid"){
			solidVoxelization = true;
		}
	}
	if (!filegiven) {
		fprintf(stdout, "[Err] You didn't specify a file using -f (path). This is required. Exiting. \n");
		printExample(); exit(1);
	}
	fprintf(stdout, "[Info] Filename: %s \n", filename.c_str());
	fprintf(stdout, "[Info] Grid size: %i \n", gridsize);
	fprintf(stdout, "[Info] Output format: %s \n", OutputFormats[int(outputformat)]);
	fprintf(stdout, "[Info] Using CPU-based voxelization: %s (default: No)\n", forceCPU ? "Yes" : "No");
	fprintf(stdout, "[Info] Using Solid Voxelization: %s (default: No)\n", solidVoxelization ? "Yes" : "No");
}

int main(int argc, char* argv[]) {
	// PRINT PROGRAM INFO
	Timer t; t.start();
	printHeader();

	// PARSE PROGRAM PARAMETERS
	fprintf(stdout, "\n## PROGRAM PARAMETERS \n");
	parseProgramParameters(argc, argv);
	fflush(stdout);
	trimesh::TriMesh::set_verbose(false);

	// READ THE MESH
	fprintf(stdout, "\n## READ MESH \n");
#ifdef _DEBUG
	trimesh::TriMesh::set_verbose(true);
#endif
	fprintf(stdout, "[I/O] Reading mesh from %s \n", filename.c_str());
	trimesh::TriMesh* themesh = trimesh::TriMesh::read(filename.c_str());
	themesh->need_faces(); // Trimesh: Unpack (possible) triangle strips so we have faces for sure
	fprintf(stdout, "[Mesh] Number of triangles: %zu \n", themesh->faces.size());
	fprintf(stdout, "[Mesh] Number of vertices: %zu \n", themesh->vertices.size());
	fprintf(stdout, "[Mesh] Computing bbox \n");
	themesh->need_bbox(); // Trimesh: Compute the bounding box (in model coordinates)

	// COMPUTE BOUNDING BOX AND VOXELISATION PARAMETERS
	fprintf(stdout, "\n## VOXELISATION SETUP \n");
	// Initialize our own AABox, pad it so it's a cube
	AABox<float3> bbox_mesh_cubed = createMeshBBCube<float3>(AABox<float3>(trimesh_to_float3(themesh->bbox.min), trimesh_to_float3(themesh->bbox.max)));
	// Create voxinfo struct and print all info
	voxinfo voxelization_info(bbox_mesh_cubed, make_uint3(gridsize, gridsize, gridsize), themesh->faces.size());
	voxelization_info.print();
	// Compute space needed to hold voxel table (1 voxel / bit)
	unsigned int* vtable = 0; // Both voxelization paths (GPU and CPU) need this
	size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(voxelization_info.gridsize.x) * static_cast<size_t>(voxelization_info.gridsize.y) * static_cast<size_t>(voxelization_info.gridsize.z) / 32.0f) * 4);

	// CUDA initialization
	bool cuda_ok = false;
	if (!forceCPU)
	{
		// SECTION: Try to figure out if we have a CUDA-enabled GPU
		fprintf(stdout, "\n## CUDA INIT \n");
		cuda_ok = initCuda();
		if (! cuda_ok ) fprintf(stdout, "[Info] CUDA GPU not found\n");
	}

	// SECTION: The actual voxelization
	if (cuda_ok && !forceCPU) { 
		// GPU voxelization
		fprintf(stdout, "\n## TRIANGLES TO GPU TRANSFER \n");

		float* device_triangles;

		// Transfer triangle data to GPU
		device_triangles = meshToGPU_managed(themesh);

		// Allocate memory for voxel grid
		fprintf(stdout, "[Voxel Grid] Allocating %s of CUDA-managed UNIFIED memory for Voxel Grid\n", readableSize(vtable_size).c_str());
		checkCudaErrors(cudaMallocManaged((void**)&vtable, vtable_size));
		
		fprintf(stdout, "\n## GPU VOXELISATION \n");
		if (solidVoxelization){
			voxelize_solid(voxelization_info, device_triangles, vtable, (outputformat == OutputFormat::output_morton));
		}
		else{
			voxelize(voxelization_info, device_triangles, vtable, (outputformat == OutputFormat::output_morton));
		}
	} else { 
		// CPU VOXELIZATION FALLBACK
		fprintf(stdout, "\n## CPU VOXELISATION \n");
		if (!forceCPU) { fprintf(stdout, "[Info] No suitable CUDA GPU was found: Falling back to CPU voxelization\n"); }
		else { fprintf(stdout, "[Info] Doing CPU voxelization (forced using command-line switch -cpu)\n"); }
		// allocate zero-filled array
		vtable = (unsigned int*) calloc(1, vtable_size);
		if (!solidVoxelization) {
			cpu_voxelizer::cpu_voxelize_mesh(voxelization_info, themesh, vtable, (outputformat == OutputFormat::output_morton));
		}
		else {
			cpu_voxelizer::cpu_voxelize_mesh_solid(voxelization_info, themesh, vtable, (outputformat == OutputFormat::output_morton));
		}
	}

	//// DEBUG: print vtable
	//for (int i = 0; i < vtable_size; i++) {
	//	char* vtable_p = (char*)vtable;
	//	cout << (int) vtable_p[i] << endl;
	//}

	fprintf(stdout, "\n## FILE OUTPUT \n");
	if (outputformat == OutputFormat::output_morton){
		write_binary(vtable, vtable_size, filename);
	} else if (outputformat == OutputFormat::output_binvox){
		write_binvox(vtable, voxelization_info, filename);
	}
	else if (outputformat == OutputFormat::output_obj_points) {
		write_obj_pointcloud(vtable, voxelization_info, filename);
	}
	else if (outputformat == OutputFormat::output_obj_cubes) {
		write_obj_cubes(vtable, voxelization_info, filename);
	}
	else if (outputformat == OutputFormat::output_vox) {
		write_vox(vtable, voxelization_info, filename);
	}

	fprintf(stdout, "\n## STATS \n");
	t.stop(); fprintf(stdout, "[Perf] Total runtime: %.1f ms \n", t.elapsed_time_milliseconds);
}