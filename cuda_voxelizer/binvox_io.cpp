#include "binvox_io.h"

using namespace std;



char checkVoxel(size_t x, size_t y, size_t z, size_t gridsize, const unsigned int* vtable){
	size_t location = x + (y*gridsize) + (z*gridsize*gridsize);
	size_t int_location = int(location / 32.0f);
	unsigned int bit_pos = 31 - unsigned int(int(location) % 32); // we count bit positions RtL, but array indices LtR
	if ((vtable[int_location]) & (1 << bit_pos)){
		return char (1);
	}
	return char(0);
}

void write_binvox(const unsigned int* vtable, size_t gridsize, std::string filename){

	//size_t voxel_count = 0;
	//for (size_t x = 0; x < gridsize; x++){
	//	for (size_t z = 0; z < gridsize; z++){
	//		for (size_t y = 1; y < gridsize; y++){
	//			if (checkVoxel(x, y, z, gridsize, vtable) == 1){
	//				voxel_count++;
	//			}
	//		}
	//	}
	//}

	//fprintf(stdout, "I think there are %llu voxels in the vtable \n", voxel_count);

	// Open file
	ofstream output(filename.c_str(), ios::out | ios::binary);
	
	// Write ASCII header
	output << "#binvox 1" << endl;
	output << "dim " << gridsize << " " << gridsize << " " << gridsize << "" << endl;
	output << "data" << endl;


	// Write first voxel
	char currentvalue = checkVoxel(0, 0, 0, gridsize, vtable);
	output.write((char*)&currentvalue, 1);
	char current_seen = 1;

	// Write BINARY Data
	for (size_t x = 0; x < gridsize; x++){
		for (size_t z = 0; z < gridsize; z++){
			for (size_t y = 0; y < gridsize; y++){
				if (x == 0 && y == 0 && z == 0){
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