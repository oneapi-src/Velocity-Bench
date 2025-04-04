// Modifications Copyright (C) 2023 Intel Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
// OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
// OR OTHER DEALINGS IN THE SOFTWARE.

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include "util.h"
#include "util_io.h"
#include "TriMesh_algo.h"
#include <cmath>

using namespace std;

// helper function to get file length (in number of ASCII characters)
size_t get_file_length(const std::string base_filename)
{
	// open file at the end
	std::ifstream input(base_filename.c_str(), ios_base::ate | ios_base::binary);
	assert(input);
	size_t length = input.tellg();
	input.close();
	return length; // get file length
}

// read raw bytes from file
void read_binary(void *data, const size_t length, const std::string base_filename)
{
	// open file
	std::ifstream input(base_filename.c_str(), ios_base::in | ios_base::binary);
	assert(input);
#ifndef SILENT
	fprintf(stdout, "[I/O] Reading %llu kb of binary data from file %s \n", size_t(length / 1024.0f), base_filename.c_str());
	fflush(stdout);
#endif
	input.seekg(0, input.beg);
	input.read((char *)data, 8);
	input.close();
	return;
}

// Helper function to write single vertex normal to OBJ file
static void write_vertex_normal(ofstream &output, const sycl::int3 &v)
{
	output << "vn " << v.x() << " " << v.y() << " " << v.z() << endl;
}

// Helper function to write single vertex to OBJ file
static void write_vertex(ofstream &output, const sycl::int3 &v)
{
	output << "v " << v.x() << " " << v.y() << " " << v.z() << endl;
}

// Helper function to write single vertex
static void write_face(ofstream &output, const sycl::int3 &v)
{
	output << "f " << v.x() << " " << v.y() << " " << v.z() << endl;
}

// Helper function to write full cube (using relative vertex positions in the OBJ file - support for this should be widespread by now)
void write_cube(const size_t &x, const size_t &y, const size_t &z, ofstream &output)
{
	//	   2-------1
	//	  /|      /|
	//	 / |     / |
	//	7--|----8  |
	//	|  4----|--3
	//	| /     | /
	//	5-------6
	// Create vertices
	sycl::int3 v1(x + 1, y + 1, z + 1);
	sycl::int3 v2(x, y + 1, z + 1);
	sycl::int3 v3(x + 1, y, z + 1);
	sycl::int3 v4(x, y, z + 1);
	sycl::int3 v5(x, y, z);
	sycl::int3 v6(x + 1, y, z);
	sycl::int3 v7(x, y + 1, z);
	sycl::int3 v8(x + 1, y + 1, z);
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
	write_face(output, sycl::int3(-1, -3, -4));
	write_face(output, sycl::int3(-1, -4, -2));
	// bottom
	write_face(output, sycl::int3(-4, -3, -6));
	write_face(output, sycl::int3(-4, -6, -5));
	// right
	write_face(output, sycl::int3(-3, -1, -8));
	write_face(output, sycl::int3(-3, -8, -6));
	// top
	write_face(output, sycl::int3(-1, -2, -7));
	write_face(output, sycl::int3(-1, -7, -8));
	// left
	write_face(output, sycl::int3(-2, -4, -5));
	write_face(output, sycl::int3(-2, -5, -7));
	// front
	write_face(output, sycl::int3(-5, -6, -8));
	write_face(output, sycl::int3(-5, -8, -7));
}

void write_obj_cubes(const unsigned int *vtable, const voxinfo v_info, const std::string base_filename)
{
	string filename_output = base_filename + string("_") + to_string(v_info.gridsize.x()) + string("_voxels.obj");
#ifndef SILENT
	fprintf(stdout, "[I/O] Writing data in obj voxels format to file %s \n", filename_output.c_str());
#endif
	ofstream output(filename_output.c_str(), ios::out);

	// Write vertex normals once
	// write_vertex_normal(output, sycl::int3(0, 0, -1)); // forward = 1
	// write_vertex_normal(output, sycl::int3(0, 0, 1)); // backward = 2
	// write_vertex_normal(output, sycl::int3(-1, 0, 0)); // left = 3
	// write_vertex_normal(output, sycl::int3(1, 0, 0)); // right = 4
	// write_vertex_normal(output, sycl::int3(0, -1, 0)); // bottom = 5
	// write_vertex_normal(output, sycl::int3(0, 1, 0)); // top = 6

	// Write stats
	size_t voxels_seen = 0;
	const size_t write_stats_25 = (size_t(v_info.gridsize.x()) * size_t(v_info.gridsize.y()) * size_t(v_info.gridsize.z())) / 4.0f;
	fprintf(stdout, "[I/O] Writing to file: 0%%...");

	// size_t voxels_written = 0;
	assert(output);
	for (size_t x = 0; x < v_info.gridsize.x(); x++)
	{
		for (size_t y = 0; y < v_info.gridsize.y(); y++)
		{
			for (size_t z = 0; z < v_info.gridsize.z(); z++)
			{
				voxels_seen++;
				if (voxels_seen == write_stats_25)
				{
					fprintf(stdout, "25%%...");
				}
				else if (voxels_seen == write_stats_25 * size_t(2))
				{
					fprintf(stdout, "50%%...");
				}
				else if (voxels_seen == write_stats_25 * size_t(3))
				{
					fprintf(stdout, "75%%...");
				}
				if (checkVoxel(x, y, z, v_info.gridsize, vtable))
				{
					// voxels_written += 1;
					write_cube(x, y, z, output);
				}
			}
		}
	}
	fprintf(stdout, "100%% \n");
	// std::cout << "written " << voxels_written << std::endl;

	fprintf(stdout, "[I/O] Reordering / Optimizing mesh with Trimesh2 \n");
	// Load the file using TriMesh2
	trimesh::TriMesh *temp_mesh = trimesh::TriMesh::read(filename_output.c_str());
	trimesh::reorder_verts(temp_mesh);
	// trimesh::faceflip(temp_mesh);
	// trimesh::edgeflip(temp_mesh);
	// temp_mesh->clear_normals();
	// temp_mesh->need_normals();
	fprintf(stdout, "[I/O] Writing final mesh to file %s \n", filename_output.c_str());
	temp_mesh->write(filename_output.c_str());

	output.close();
}

void write_obj_pointcloud(const unsigned int *vtable, const voxinfo v_info, const std::string base_filename)
{
	string filename_output = base_filename + string("_") + to_string(v_info.gridsize.x()) + string("_pointcloud.obj");
#ifndef SILENT
	fprintf(stdout, "[I/O] Writing data in obj point cloud format to %s \n", filename_output.c_str());
#endif
	ofstream output(filename_output.c_str(), ios::out);

	// Write stats
	size_t voxels_seen = 0;
	const size_t write_stats_25 = (size_t(v_info.gridsize.x()) * size_t(v_info.gridsize.y()) * size_t(v_info.gridsize.z())) / 4.0f;
	fprintf(stdout, "[I/O] Writing to file: 0%%...");

	// write stats
	size_t voxels_written = 0;

	assert(output);
	for (size_t x = 0; x < v_info.gridsize.x(); x++)
	{
		for (size_t y = 0; y < v_info.gridsize.y(); y++)
		{
			for (size_t z = 0; z < v_info.gridsize.z(); z++)
			{
				voxels_seen++;
				if (voxels_seen == write_stats_25)
				{
					fprintf(stdout, "25%%...");
				}
				else if (voxels_seen == write_stats_25 * size_t(2))
				{
					fprintf(stdout, "50%%...");
				}
				else if (voxels_seen == write_stats_25 * size_t(3))
				{
					fprintf(stdout, "75%%...");
				}
				if (checkVoxel(x, y, z, v_info.gridsize, vtable))
				{
					voxels_written += 1;
					output << "v " << (x + 0.5) << " " << (y + 0.5) << " " << (z + 0.5) << endl; // +0.5 to put vertex in the middle of the voxel
				}
			}
		}
	}
	fprintf(stdout, "100%% \n");
	// std::cout << "written " << voxels_written << std::endl;
	output.close();
}

void write_binary(void *data, size_t bytes, const std::string base_filename)
{
	string filename_output = base_filename + string(".bin");
#ifndef SILENT
	fprintf(stdout, "[I/O] Writing data in binary format to %s (%s) \n", filename_output.c_str(), readableSize(bytes).c_str());
#endif
	ofstream output(filename_output.c_str(), ios_base::out | ios_base::binary);
	output.write((char *)data, bytes);
	output.close();
}

void write_binvox(const unsigned int *vtable, const voxinfo v_info, const std::string base_filename)
{
	// Open file
	string filename_output = base_filename + string("_") + to_string(v_info.gridsize.x()) + string(".binvox");
#ifndef SILENT
	fprintf(stdout, "[I/O] Writing data in binvox format to %s \n", filename_output.c_str());
#endif
	ofstream output(filename_output.c_str(), ios::out | ios::binary);
	assert(output);
	// Write ASCII header
	output << "#binvox 1" << endl;
	output << "dim " << v_info.gridsize.x() << " " << v_info.gridsize.y() << " " << v_info.gridsize.z() << "" << endl;
	output << "translate " << v_info.bbox.min.x() << " " << v_info.bbox.min.y() << " " << v_info.bbox.min.z() << endl;
	output << "scale "
		   << max(fmaxf(v_info.bbox.max.x() - v_info.bbox.min.x(),
						v_info.bbox.max.y() - v_info.bbox.min.y()),
				  v_info.bbox.max.z() - v_info.bbox.min.z())
		   << endl;
	output << "data" << endl;

	// Write BINARY Data (and compress it a bit using run-length encoding)
	char currentvalue, current_seen;
	for (size_t x = 0; x < v_info.gridsize.x(); x++)
	{
		for (size_t z = 0; z < v_info.gridsize.z(); z++)
		{
			for (size_t y = 0; y < v_info.gridsize.y(); y++)
			{
				if (x == 0 && y == 0 && z == 0)
				{ // special case: first voxel
					currentvalue = checkVoxel(0, 0, 0, v_info.gridsize, vtable);
					output.write((char *)&currentvalue, 1);
					current_seen = 1;
					continue;
				}
				char nextvalue = checkVoxel(x, y, z, v_info.gridsize, vtable);
				if (nextvalue != currentvalue || current_seen == (char)255)
				{
					output.write((char *)&current_seen, 1);
					current_seen = 1;
					currentvalue = nextvalue;
					output.write((char *)&currentvalue, 1);
				}
				else
				{
					current_seen++;
				}
			}
		}
	}

	// Write rest
	output.write((char *)&current_seen, 1);
	output.close();
}