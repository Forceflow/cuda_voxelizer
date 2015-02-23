#ifndef BINVOX_IO_H_
#define BINVOX_IO_H_

#include <string>
#include <iostream>
#include <fstream>
#include "util.h"

void write_binary(const void* data, size_t bytes, std::string base_filename);
void write_binvox(const unsigned int* vtable, size_t gridsize, std::string base_filename);

#endif