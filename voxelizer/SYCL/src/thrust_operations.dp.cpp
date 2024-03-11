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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include "thrust_operations.dp.hpp"

std::vector<sycl::float3> *trianglethrust_host;
infra::device_vector<sycl::float3> *trianglethrust_device;

// method 3: use a thrust vector
float *meshToGPU_thrust(const trimesh::TriMesh *mesh)
{
	Timer t;
	t.start(); // TIMER START
			   // create vectors on heap
	trianglethrust_host = new std::vector<sycl::float3>;
	trianglethrust_device = new infra::device_vector<sycl::float3>;
	// fill host vector
	fprintf(stdout, "[Mesh] Copying %zu triangles to Thrust host vector \n", mesh->faces.size());
	for (size_t i = 0; i < mesh->faces.size(); i++)
	{
		sycl::float3 v0 = trimesh_to_sycl<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
		sycl::float3 v1 = trimesh_to_sycl<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
		sycl::float3 v2 = trimesh_to_sycl<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
		trianglethrust_host->push_back(v0);
		trianglethrust_host->push_back(v1);
		trianglethrust_host->push_back(v2);
	}
	fprintf(stdout, "[Mesh] Copying Thrust host vector to Thrust device vector \n");
	*trianglethrust_device = *trianglethrust_host;
	t.stop();
	fprintf(stdout, "[Mesh] Transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds); // TIMER END
	return (float *)infra::get_raw_pointer(&((*trianglethrust_device)[0]));
}

void cleanup_thrust()
{
	fprintf(stdout, "[Mesh] Freeing Thrust host and device vectors \n");
	if (trianglethrust_device)
		free(trianglethrust_device);
	if (trianglethrust_host)
		free(trianglethrust_host);
}