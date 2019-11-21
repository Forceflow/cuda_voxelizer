#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include "assert.h"

size_t get_file_length(const std::string base_filename);
void read_binary(void* data, const size_t length, const std::string base_filename);
void write_binary(void* data, const size_t bytes, const std::string base_filename);
void write_binvox(const unsigned int* vtable, const size_t gridsize, const std::string base_filename);
void write_obj(const unsigned int* vtable, const size_t gridsize, const std::string base_filename);
