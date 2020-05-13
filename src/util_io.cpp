#include "util.h"
#include "util_io.h"

using namespace std;

// helper function to get file length (in number of ASCII characters)
size_t get_file_length(const std::string base_filename){
	// open file at the end
	std::ifstream input(base_filename.c_str(), ios_base::ate | ios_base::binary);
	assert(input);
	size_t length = input.tellg();
	input.close();
	return length; // get file length
}

// read raw bytes from file
void read_binary(void* data, const size_t length, const std::string base_filename){
	// open file
	std::ifstream input(base_filename.c_str(), ios_base::in | ios_base::binary);
	assert(input);
#ifndef SILENT
	fprintf(stdout, "[I/O] Reading %llu kb of binary data from file %s \n", size_t(length / 1024.0f), base_filename.c_str()); fflush(stdout);
#endif
	input.seekg(0, input.beg);
	input.read((char*) data, 8);
	input.close();
	return;
}

// Helper function to write single vertex normal to OBJ file
void write_vertex_normal(ofstream& output, const glm::ivec3& v) {
	output << "vn " << v.x << " " << v.y << " " << v.z << endl;
}

// Helper function to write single vertex to OBJ file
void write_vertex(ofstream& output, const glm::ivec3& v) {
	output << "v " << v.x << " " << v.y << " " << v.z << endl;
}

//// Helper function to write single vertex
//void write_face(ofstream& output, const glm::ivec3& v, const glm::ivec3& n) {
//	output << "f " << v.x << "//" << n.x << " " << v.y << "//" << n.y << " " << v.z << "//" << n.z << endl;
//}

// Helper function to write single vertex
void write_face(ofstream& output, const glm::ivec3& v) {
	output << "f " << v.x << " " << v.y << " " << v.z << endl;
}

// Helper function to write full cube (using relative vertex positions in the OBJ file - support for this should be widespread by now)
void write_cube(const size_t& x, const size_t& y, const size_t& z, ofstream& output) {
	//	   2-------1
	//	  /|      /|
	//	 / |     / |
	//	7--|----8  |
	//	|  4----|--3
	//	| /     | /
	//	5-------6
    // Create vertices
	glm::ivec3 v1(x+1, y+1, z + 1);
	glm::ivec3 v2(x, y+1, z + 1);
	glm::ivec3 v3(x+1, y, z + 1);
	glm::ivec3 v4(x, y, z + 1);
	glm::ivec3 v5(x, y, z);
	glm::ivec3 v6(x+1, y, z);
	glm::ivec3 v7(x, y+1, z);
	glm::ivec3 v8(x+1, y+1, z);
	// write them in reverse order, so relative position is -i for v_i
	write_vertex(output, v8);
	write_vertex(output, v7);
	write_vertex(output, v6);
	write_vertex(output, v5);
	write_vertex(output, v4);
	write_vertex(output, v3);
	write_vertex(output, v2);
	write_vertex(output, v1);
	// create faces
	// back
	write_face(output, glm::ivec3(-1, -3, -4));
	write_face(output, glm::ivec3(-1, -4, -2));
	// bottom
	write_face(output, glm::ivec3(-4, -3, -6));
	write_face(output, glm::ivec3(-4, -6, -5));
	// right
	write_face(output, glm::ivec3(-3, -1, -8));
	write_face(output, glm::ivec3(-3, -8, -6));
	// top
	write_face(output, glm::ivec3(-1, -2, -7));
	write_face(output, glm::ivec3(-1, -7, -8));
	// left
	write_face(output, glm::ivec3(-2, -4, -5));
	write_face(output, glm::ivec3(-2, -5, -7));
	// front
	write_face(output, glm::ivec3(-5, -6, -8));
	write_face(output, glm::ivec3(-5, -8, -7));
}

void write_obj_cubes(const unsigned int* vtable, const size_t gridsize, const std::string base_filename) {
	string filename_output = base_filename + string("_") + to_string(gridsize) + string("_voxels.obj");
#ifndef SILENT
	fprintf(stdout, "[I/O] Writing data in obj voxels format to %s \n", filename_output.c_str());
#endif
	ofstream output(filename_output.c_str(), ios::out);

	// Write vertex normals once
	//write_vertex_normal(output, glm::ivec3(0, 0, -1)); // forward = 1
	//write_vertex_normal(output, glm::ivec3(0, 0, 1)); // backward = 2
	//write_vertex_normal(output, glm::ivec3(-1, 0, 0)); // left = 3
	//write_vertex_normal(output, glm::ivec3(1, 0, 0)); // right = 4
	//write_vertex_normal(output, glm::ivec3(0, -1, 0)); // bottom = 5
	//write_vertex_normal(output, glm::ivec3(0, 1, 0)); // top = 6

	size_t voxels_written = 0;
	assert(output);
	for (size_t x = 0; x < gridsize; x++) {
		for (size_t y = 0; y < gridsize; y++) {
			for (size_t z = 0; z < gridsize; z++) {
				if (checkVoxel(x, y, z, gridsize, vtable)) {
					voxels_written += 1;
					write_cube(x, y, z, output);
				}
			}
		}
	}
	std::cout << "written " << voxels_written << std::endl;
	output.close();
}

void write_obj_pointcloud(const unsigned int* vtable, const size_t gridsize, const std::string base_filename) {
	string filename_output = base_filename + string("_") + to_string(gridsize) + string("_pointcloud.obj");
#ifndef SILENT
	fprintf(stdout, "[I/O] Writing data in obj point cloud format to %s \n", filename_output.c_str());
#endif
	ofstream output(filename_output.c_str(), ios::out);
	size_t voxels_written = 0;
	assert(output);
	for (size_t x = 0; x < gridsize; x++) {
		for (size_t y = 0; y < gridsize; y++) {
			for (size_t z = 0; z < gridsize; z++) {
				if (checkVoxel(x, y, z, gridsize, vtable)) {
					voxels_written += 1;
					output << "v " << (x+0.5) << " " << (y + 0.5) << " " << (z + 0.5) << endl; // +0.5 to put vertex in the middle of the voxel
				}
			}
		}
	}
	std::cout << "written " << voxels_written << std::endl;
	output.close();
}

void write_binary(void* data, size_t bytes, const std::string base_filename){
	string filename_output = base_filename + string(".bin");
#ifndef SILENT
	fprintf(stdout, "[I/O] Writing data in binary format to %s (%s) \n", filename_output.c_str(), readableSize(bytes).c_str());
#endif
	ofstream output(filename_output.c_str(), ios_base::out | ios_base::binary);
	output.write((char*)data, bytes);
	output.close();
}

void write_binvox(const unsigned int* vtable, const size_t gridsize, const std::string base_filename){
	// Open file
	string filename_output = base_filename + string("_") + to_string(gridsize) + string(".binvox");
#ifndef SILENT
	fprintf(stdout, "[I/O] Writing data in binvox format to %s \n", filename_output.c_str());
#endif
	ofstream output(filename_output.c_str(), ios::out | ios::binary);
	assert(output);
	
	// Write ASCII header
	output << "#binvox 1" << endl;
	output << "dim " << gridsize << " " << gridsize << " " << gridsize << "" << endl;
	output << "data" << endl;

	// Write BINARY Data (and compress it a bit using run-length encoding)
	char currentvalue, current_seen;
	for (size_t x = 0; x < gridsize; x++){
		for (size_t z = 0; z < gridsize; z++){
			for (size_t y = 0; y < gridsize; y++){
				if (x == 0 && y == 0 && z == 0){ // special case: first voxel
					currentvalue = checkVoxel(0, 0, 0, gridsize, vtable);
					output.write((char*)&currentvalue, 1);
					current_seen = 1;
					continue;
				}
				char nextvalue = checkVoxel(x, y, z, gridsize, vtable);
				if (nextvalue != currentvalue || current_seen == (char) 255){
					output.write((char*)&current_seen, 1);
					current_seen = 1;
					currentvalue = nextvalue;
					output.write((char*)&currentvalue, 1);
				}
				else {
					current_seen++;
				}
			}
		}
	}

	// Write rest
	output.write((char*)&current_seen, 1);
	output.close();
}