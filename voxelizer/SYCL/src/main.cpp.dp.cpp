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

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN // Please, not too much windows shenanigans
#endif

// Standard libs
#include <CL/sycl.hpp>
#include <string>
#include <cstdio>
#include <cmath>
// Trimesh for model importing
#include "TriMesh.h"
// Util
#include "util.h"
#include "util_io.h"
#include "util_cuda.h"
#include "timer.h"
// CPU voxelizer fallback
#include "cpu_voxelizer.h"
#include "voxelize.dp.hpp"

sycl::property_list q_prop{sycl::ext::oneapi::property::queue::discard_events(), sycl::property::queue::in_order()};
sycl::queue sycl_device_queue(q_prop);
float total_gpu_time;

using namespace std;

// Forward declaration of CUDA functions
float *meshToGPU_thrust(const trimesh::TriMesh *mesh); // METHOD 3 to transfer triangles can be found in thrust_operations.cu(h)
void cleanup_thrust();
void voxelize(const voxinfo &v, float *triangle_data, unsigned int *vtable, bool useThrustPath, bool morton_code);
void voxelize_solid(const voxinfo &v, float *triangle_data, unsigned int *vtable, bool useThrustPath, bool morton_code);

// Output formats
enum class OutputFormat
{
	output_binvox = 0,
	output_morton = 1,
	output_obj_points = 2,
	output_obj_cubes = 3
};
char *OutputFormats[] = {"binvox file", "morton encoded blob", "obj file (pointcloud)", "obj file (cubes)"};

// Default options
string filename = "";
string filename_base = "";
OutputFormat outputformat = OutputFormat::output_binvox;
unsigned int gridsize = 256;
int iterations = 1;
bool useThrustPath = false;
bool forceCPU = false;
bool solidVoxelization = false;

void printHeader()
{
	fprintf(stdout, "## VOXELIZER \n");
}

void printExample()
{
	cout << "Example: voxelizer_sycl -f /home/usr/bunny.ply -s 512" << endl;
}

void printHelp()
{
	fprintf(stdout, "\n## HELP  \n");
	cout << "Program options: " << endl
		 << endl;
	cout << " -f <path to model file: .ply, .obj, .3ds> (required)" << endl;
	cout << " -s <voxelization grid size, power of 2: 8 -> 512, 1024, ... (default: 256)>" << endl;
	cout << " -i <number of iterations (default: 1)>" << endl;
	cout << " -o <output format: binvox, obj, obj_points or morton (default: binvox)>" << endl;
	cout << " -thrust : Force using CUDA Thrust Library (possible speedup / throughput improvement)" << endl;
	cout << " -cpu : Force CPU-based voxelization (slow, but works if no compatible GPU can be found)" << endl;
	cout << " -solid : Force solid voxelization (experimental, needs watertight model)" << endl
		 << endl;
	printExample();
	cout << endl;
}

// METHOD 1: Helper function to transfer triangles to automatically managed CUDA memory ( > CUDA 7.x)
float *meshToGPU_managed(const trimesh::TriMesh *mesh)
{
	Timer t;
	t.start();
	size_t n_floats = sizeof(float) * 9 * (mesh->faces.size());
	float *device_triangles;
	fprintf(stdout, "[Mesh] Allocating %s of managed UNIFIED memory for triangle data \n", (readableSize(n_floats)).c_str());
	auto start_gpu_time = std::chrono::steady_clock::now();
	device_triangles = (float *)sycl::malloc_shared(
		n_floats, sycl_device_queue); // managed memory
	auto stop_gpu_time = std::chrono::steady_clock::now();
	total_gpu_time += std::chrono::duration<float, std::milli>(stop_gpu_time - start_gpu_time).count();

	fprintf(stdout, "[Mesh] Copy %llu triangles to managed UNIFIED memory \n", (size_t)(mesh->faces.size()));
	for (size_t i = 0; i < mesh->faces.size(); i++)
	{
		sycl::float3 v0 = trimesh_to_sycl<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
		sycl::float3 v1 = trimesh_to_sycl<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
		sycl::float3 v2 = trimesh_to_sycl<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
		size_t j = i * 9;
		memcpy((device_triangles) + j, &(v0.x()), sizeof(sycl::float3));
		memcpy((device_triangles) + j + 3, &(v1.x()), sizeof(sycl::float3));
		memcpy((device_triangles) + j + 6, &(v2.x()), sizeof(sycl::float3));
	}
	t.stop();
	fprintf(stdout, "[Perf] Mesh transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds);
	return device_triangles;
}

// METHOD 2: Helper function to transfer triangles to old-style, self-managed CUDA memory ( < CUDA 7.x )
// Leaving this here for reference, the function above should be faster and better managed on all versions CUDA 7+
//
// float* meshToGPU(const trimesh::TriMesh *mesh){
//	size_t n_floats = sizeof(float) * 9 * (mesh->faces.size());
//	float* pagelocktriangles;
//	fprintf(stdout, "Allocating %llu kb of page-locked HOST memory \n", (size_t)(n_floats / 1024.0f));
//	checkCudaErrors(cudaHostAlloc((void**)&pagelocktriangles, n_floats, cudaHostAllocDefault)); // pinned memory to easily copy from
//	fprintf(stdout, "Copy %llu triangles to page-locked HOST memory \n", (size_t)(mesh->faces.size()));
//	for (size_t i = 0; i < mesh->faces.size(); i++){
//		sycl::float3 v0 = trimesh_to_sycl<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
//		sycl::float3 v1 = trimesh_to_sycl<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
//		sycl::float3 v2 = trimesh_to_sycl<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
//		size_t j = i * 9;
//		memcpy((pagelocktriangles)+j, glm::value_ptr(v0), sizeof(sycl::float3));
//		memcpy((pagelocktriangles)+j+3, glm::value_ptr(v1), sizeof(sycl::float3));
//		memcpy((pagelocktriangles)+j+6, glm::value_ptr(v2), sizeof(sycl::float3));
//	}
//	float* device_triangles;
//	fprintf(stdout, "Allocating %llu kb of DEVICE memory \n", (size_t)(n_floats / 1024.0f));
//	checkCudaErrors(cudaMalloc((void **) &device_triangles, n_floats));
//	fprintf(stdout, "Copy %llu triangles from page-locked HOST memory to DEVICE memory \n", (size_t)(mesh->faces.size()));
//	checkCudaErrors(cudaMemcpy((void *) device_triangles, (void*) pagelocktriangles, n_floats, cudaMemcpyDefault));
//	return device_triangles;
//}

// Parse the program parameters and set them as global variables
void parseProgramParameters(int argc, char *argv[])
{
	if (argc < 2)
	{ // not enough arguments
		fprintf(stdout, "Not enough program parameters. \n \n");
		printHelp();
		exit(0);
	}
	bool filegiven = false;
	for (int i = 1; i < argc; i++)
	{
		if (string(argv[i]) == "-f")
		{
			filename = argv[i + 1];
			filename_base = filename.substr(0, filename.find_last_of("."));
			filegiven = true;
			if (!file_exists(filename))
			{
				fprintf(stdout, "[Err] File does not exist / cannot access: %s \n", filename.c_str());
				exit(1);
			}
			i++;
		}
		else if (string(argv[i]) == "-s")
		{
			gridsize = atoi(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "-i")
		{
			iterations = atoi(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "-h")
		{
			printHelp();
			exit(0);
		}
		else if (string(argv[i]) == "-o")
		{
			string output = (argv[i + 1]);
			transform(output.begin(), output.end(), output.begin(), ::tolower); // to lowercase
			if (output == "binvox")
			{
				outputformat = OutputFormat::output_binvox;
			}
			else if (output == "morton")
			{
				outputformat = OutputFormat::output_morton;
			}
			else if (output == "obj")
			{
				outputformat = OutputFormat::output_obj_cubes;
			}
			else if (output == "obj_points")
			{
				outputformat = OutputFormat::output_obj_points;
			}
			else
			{
				fprintf(stdout, "[Err] Unrecognized output format: %s, valid options are binvox (default), morton, obj or obj_points \n", output.c_str());
				exit(1);
			}
		}
		else if (string(argv[i]) == "-thrust")
		{
			useThrustPath = true;
		}
		else if (string(argv[i]) == "-cpu")
		{
			forceCPU = true;
		}
		else if (string(argv[i]) == "-solid")
		{
			solidVoxelization = true;
		}
	}
	if (!filegiven)
	{
		fprintf(stdout, "[Err] You didn't specify a file using -f (path). This is required. Exiting. \n");
		printExample();
		exit(1);
	}
	fprintf(stdout, "[Info] Filename: %s \n", filename.c_str());
	fprintf(stdout, "[Info] Grid size: %i \n", gridsize);
	fprintf(stdout, "[Info] Iterations: %i (default : 1)\n", iterations);
	fprintf(stdout, "[Info] Output format: %s \n", OutputFormats[int(outputformat)]);
	fprintf(stdout, "[Info] Using CUDA Thrust: %s (default: No)\n", useThrustPath ? "Yes" : "No");
	fprintf(stdout, "[Info] Using CPU-based voxelization: %s (default: No)\n", forceCPU ? "Yes" : "No");
	fprintf(stdout, "[Info] Using Solid Voxelization: %s (default: No)\n", solidVoxelization ? "Yes" : "No");
}

int main(int argc, char *argv[])
{

	auto totalProgTimer_start = std::chrono::steady_clock::now();
	total_gpu_time = 0.0;
	// HOIST INTO SETUP FUNCTION EVENTUALLY
	sycl_device_queue = sycl::gpu_selector{};

	// PRINT PROGRAM INFO
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
	auto ioRead_start = std::chrono::steady_clock::now();
	trimesh::TriMesh *themesh = trimesh::TriMesh::read(filename.c_str());
	auto ioRead_stop = std::chrono::steady_clock::now();
	float ioReadTime = std::chrono::duration<float, std::micro>(ioRead_stop - ioRead_start).count();
	themesh->need_faces(); // Trimesh: Unpack (possible) triangle strips so we have faces for sure
	fprintf(stdout, "[Mesh] Number of triangles: %zu \n", themesh->faces.size());
	fprintf(stdout, "[Mesh] Number of vertices: %zu \n", themesh->vertices.size());
	fprintf(stdout, "[Mesh] Computing bbox \n");
	themesh->need_bbox(); // Trimesh: Compute the bounding box (in model coordinates)

	// COMPUTE BOUNDING BOX AND VOXELISATION PARAMETERS
	fprintf(stdout, "\n## VOXELISATION SETUP \n");
	// Initialize our own AABox, pad it so it's a cube
	AABox<sycl::float3> bbox_mesh_cubed = createMeshBBCube<sycl::float3>(AABox<sycl::float3>(trimesh_to_sycl(themesh->bbox.min), trimesh_to_sycl(themesh->bbox.max)));
	// Create voxinfo struct and print all info
	voxinfo voxelization_info(bbox_mesh_cubed, sycl::uint3(gridsize, gridsize, gridsize), themesh->faces.size());
	voxelization_info.print();
	// Compute space needed to hold voxel table (1 voxel / bit)
	unsigned int *vtable; // Both voxelization paths (GPU and CPU) need this
	size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(voxelization_info.gridsize.x()) * static_cast<size_t>(voxelization_info.gridsize.y()) * static_cast<size_t>(voxelization_info.gridsize.z()) / 32.0f) * 4);

	// CUDA initialization
	bool cuda_ok = false;
	if (!forceCPU)
	{
		// SECTION: Try to figure out if we have a CUDA-enabled GPU
		fprintf(stdout, "\n## oneAPI INIT \n");
		cuda_ok = initCuda(sycl_device_queue);
		if (!cuda_ok)
			fprintf(stdout, "[Info] GPU not found\n");
	}

	// SECTION: The actual voxelization
	for (int i = 0; i < iterations; ++i)
	{
		if (cuda_ok && !forceCPU)
		{
			// GPU voxelization
			fprintf(stdout, "\n## TRIANGLES TO GPU TRANSFER \n");

			float *device_triangles;
			// Transfer triangles to GPU using either thrust or managed cuda memory
			if (useThrustPath)
			{
				device_triangles = meshToGPU_thrust(themesh);
			}
			else
			{
				device_triangles = meshToGPU_managed(themesh);
			}

			if (!useThrustPath)
			{
				fprintf(stdout, "[Voxel Grid] Allocating %s of managed UNIFIED memory for Voxel Grid\n", readableSize(vtable_size).c_str());
				auto start_gpu_time = std::chrono::steady_clock::now();
				vtable = (unsigned int *)sycl::malloc_shared(
					vtable_size, sycl_device_queue);
				sycl_device_queue.wait();
				auto stop_gpu_time = std::chrono::steady_clock::now();
				total_gpu_time += std::chrono::duration<float, std::milli>(stop_gpu_time - start_gpu_time).count();
			}
			else
			{
				// ALLOCATE MEMORY ON HOST
				fprintf(stdout, "[Voxel Grid] Allocating %s kB of page-locked HOST memory for Voxel Grid\n", readableSize(vtable_size).c_str());
				vtable = (unsigned int *)sycl::malloc_host(
					vtable_size, sycl_device_queue);
				//  sycl_device_queue.wait();
			}
			fprintf(stdout, "\n## GPU VOXELISATION \n");
			if (solidVoxelization)
			{
				voxelize_solid(voxelization_info, device_triangles, vtable, useThrustPath, (outputformat == OutputFormat::output_morton));
			}
			else
			{
				voxelize(voxelization_info, device_triangles, vtable, useThrustPath, (outputformat == OutputFormat::output_morton));
			}
		}
		else
		{
			// CPU VOXELIZATION FALLBACK
			fprintf(stdout, "\n## CPU VOXELISATION \n");
			if (!forceCPU)
			{
				fprintf(stdout, "[Info] No suitable CUDA GPU was found: Falling back to CPU voxelization\n");
			}
			else
			{
				fprintf(stdout, "[Info] Doing CPU voxelization (forced using command-line switch -cpu)\n");
			}
			// allocate zero-filled array
			vtable = (unsigned int *)calloc(1, vtable_size);
			if (!solidVoxelization)
			{
				cpu_voxelizer::cpu_voxelize_mesh(voxelization_info, themesh, vtable, (outputformat == OutputFormat::output_morton));
			}
			else
			{
				cpu_voxelizer::cpu_voxelize_mesh_solid(voxelization_info, themesh, vtable, (outputformat == OutputFormat::output_morton));
			}
		}
	}

	//// DEBUG: print vtable
	// for (int i = 0; i < vtable_size; i++) {
	//	char* vtable_p = (char*)vtable;
	//	cout << (int) vtable_p[i] << endl;
	// }

	fprintf(stdout, "\n## FILE OUTPUT \n");
	string output_filename = filename + "_SYCL";

	auto ioWrite_start = std::chrono::steady_clock::now();
	if (outputformat == OutputFormat::output_morton)
	{
		write_binary(vtable, vtable_size, output_filename);
	}
	else if (outputformat == OutputFormat::output_binvox)
	{
		write_binvox(vtable, voxelization_info, output_filename);
	}
	else if (outputformat == OutputFormat::output_obj_points)
	{
		write_obj_pointcloud(vtable, voxelization_info, output_filename);
	}
	else if (outputformat == OutputFormat::output_obj_cubes)
	{
		write_obj_cubes(vtable, voxelization_info, output_filename);
	}
	auto ioWrite_end = std::chrono::steady_clock::now();
	float ioWriteTime = std::chrono::duration<float, std::micro>(ioWrite_end - ioWrite_start).count();

	if (useThrustPath)
	{
		cleanup_thrust();
	}

	fprintf(stdout, "\n## STATS \n");
	printf("Avg GPU time : %.1f ms\n", total_gpu_time / iterations);
	auto totalProgTimer_end = std::chrono::steady_clock::now();
	float avgWorkloadTime = std::chrono::duration<float, std::micro>(totalProgTimer_end - totalProgTimer_start).count() - ioReadTime - ioWriteTime;
	avgWorkloadTime /= iterations;
	std::cout << "Avg workload time = " << avgWorkloadTime / 1000 << " ms"
			  << "\n"
			  << std::endl;
}
