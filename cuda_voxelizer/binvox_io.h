#ifndef BINVOX_IO_H_
#define BINVOX_IO_H_

#include <string>
#include <iostream>
#include <fstream>

void write_binvox(const unsigned int* vtable, size_t gridsize, std::string filename);

#endif