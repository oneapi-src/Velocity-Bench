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
#include "voxelize.dp.hpp"
#include <chrono>

// CUDA Global Memory variables

// Debug counters for some sanity checks
#ifdef _DEBUG
__device__ size_t debug_d_n_voxels_marked = 0;
__device__ size_t debug_d_n_triangles = 0;
__device__ size_t debug_d_n_voxels_tested = 0;
#endif

// Possible optimization: buffer bitsets (for now: Disabled because too much overhead)
// struct bufferedBitSetter{
//	unsigned int* voxel_table;
//	size_t current_int_location;
//	unsigned int current_mask;
//
//	__device__ __inline__ bufferedBitSetter(unsigned int* voxel_table, size_t index) :
//		voxel_table(voxel_table), current_mask(0) {
//		current_int_location = int(index / 32.0f);
//	}
//
//	__device__ __inline__ void setBit(size_t index){
//		size_t new_int_location = int(index / 32.0f);
//		if (current_int_location != new_int_location){
//			flush();
//			current_int_location = new_int_location;
//		}
//		unsigned int bit_pos = 31 - (unsigned int)(int(index) % 32);
//		current_mask = current_mask | (1 << bit_pos);
//	}
//
//	__device__ __inline__ void flush(){
//		if (current_mask != 0){
//			atomicOr(&(voxel_table[current_int_location]), current_mask);
//		}
//	}
//};

// Possible optimization: check bit before you set it - don't need to do atomic operation if it's already set to 1
// For now: overhead, so it seems
//__device__ __inline__ bool checkBit(unsigned int* voxel_table, size_t index){
//	size_t int_location = index / size_t(32);
//	unsigned int bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
//	return ((voxel_table[int_location]) & (1 << bit_pos));
//}

// Set a bit in the giant voxel table. This involves doing an atomic operation on a 32-bit word in memory.
// Blocking other threads writing to it for a very short time
__inline__ void setBit(unsigned int *voxel_table, size_t index)
{
	size_t int_location = index / size_t(32);
	unsigned int bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
	unsigned int mask = 1 << bit_pos;
	// atomicOr(&(voxel_table[int_location]), mask);
	infra::atomic_fetch_or<unsigned int, sycl::access::address_space::generic_space>(&(voxel_table[int_location]), mask);
}

// Main triangle voxelization method
void voxelize_triangle(voxinfo info, float *triangle_data, unsigned int *voxel_table, bool morton_order,
					   sycl::nd_item<3> item_ct1, uint32_t *morton256_x,
					   uint32_t *morton256_y, uint32_t *morton256_z)
{
	size_t thread_id = item_ct1.get_local_id(2) +
					   item_ct1.get_group(2) * item_ct1.get_local_range(2);
	size_t stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

	// Common variables used in the voxelization process
	sycl::float3 delta_p(info.unit.x(), info.unit.y(), info.unit.z());
	sycl::float3 grid_max(info.gridsize.x() - 1, info.gridsize.y() - 1, info.gridsize.z() - 1); // grid max (grid runs from 0 to gridsize-1)

	while (thread_id < info.n_triangles)
	{							  // every thread works on specific triangles in its stride
		size_t t = thread_id * 9; // triangle contains 9 vertices

		// COMPUTE COMMON TRIANGLE PROPERTIES
		// Move vertices to origin using bbox
		sycl::float3 v0 = sycl::float3(triangle_data[t], triangle_data[t + 1], triangle_data[t + 2]) - info.bbox.min;
		sycl::float3 v1 = sycl::float3(triangle_data[t + 3], triangle_data[t + 4], triangle_data[t + 5]) - info.bbox.min;
		sycl::float3 v2 = sycl::float3(triangle_data[t + 6], triangle_data[t + 7], triangle_data[t + 8]) - info.bbox.min;
		// Edge vectors
		sycl::float3 e0 = v1 - v0;
		sycl::float3 e1 = v2 - v1;
		sycl::float3 e2 = v0 - v2;
		// Normal vector pointing up from the triangle
		sycl::float3 n = sycl::normalize(sycl::cross(e0, e1));

		// COMPUTE TRIANGLE BBOX IN GRID
		// Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
		AABox<sycl::float3> t_bbox_world(sycl::min(v0, sycl::min(v1, v2)), sycl::max(v0, sycl::max(v1, v2)));
		// Triangle bounding box in voxel grid coordinates is the world bounding box divided by the grid unit vector
		AABox<sycl::int3> t_bbox_grid;
		auto temp = sycl::clamp(t_bbox_world.min / info.unit, sycl::float3(0.0f, 0.0f, 0.0f), grid_max);
		t_bbox_grid.min = sycl::int3(temp.x(), temp.y(), temp.z());
		temp = sycl::clamp(t_bbox_world.max / info.unit, sycl::float3(0.0f, 0.0f, 0.0f), grid_max);
		t_bbox_grid.max = sycl::int3(temp.x(), temp.y(), temp.z());

		// PREPARE PLANE TEST PROPERTIES
		sycl::float3 c(0.0f, 0.0f, 0.0f);
		if (n.x() > 0.0f)
		{
			c.x() = info.unit.x();
		}
		if (n.y() > 0.0f)
		{
			c.y() = info.unit.y();
		}
		if (n.z() > 0.0f)
		{
			c.z() = info.unit.z();
		}
		float d1 = sycl::dot(n, (c - v0));
		float d2 = sycl::dot(n, ((delta_p - c) - v0));

		// PREPARE PROJECTION TEST PROPERTIES
		// XY plane
		sycl::float2 n_xy_e0(-1.0f * e0.y(), e0.x());
		sycl::float2 n_xy_e1(-1.0f * e1.y(), e1.x());
		sycl::float2 n_xy_e2(-1.0f * e2.y(), e2.x());
		if (n.z() < 0.0f)
		{
			n_xy_e0 = -n_xy_e0;
			n_xy_e1 = -n_xy_e1;
			n_xy_e2 = -n_xy_e2;
		}
		float d_xy_e0 = (-1.0f * sycl::dot(n_xy_e0, sycl::float2(v0.x(), v0.y()))) + sycl::max(0.0f, info.unit.x() * n_xy_e0[0]) + sycl::max(0.0f, info.unit.y() * n_xy_e0[1]);
		float d_xy_e1 = (-1.0f * sycl::dot(n_xy_e1, sycl::float2(v1.x(), v1.y()))) + sycl::max(0.0f, info.unit.x() * n_xy_e1[0]) + sycl::max(0.0f, info.unit.y() * n_xy_e1[1]);
		float d_xy_e2 = (-1.0f * sycl::dot(n_xy_e2, sycl::float2(v2.x(), v2.y()))) + sycl::max(0.0f, info.unit.x() * n_xy_e2[0]) + sycl::max(0.0f, info.unit.y() * n_xy_e2[1]);
		// YZ plane
		sycl::float2 n_yz_e0(-1.0f * e0.z(), e0.y());
		sycl::float2 n_yz_e1(-1.0f * e1.z(), e1.y());
		sycl::float2 n_yz_e2(-1.0f * e2.z(), e2.y());
		if (n.x() < 0.0f)
		{
			n_yz_e0 = -n_yz_e0;
			n_yz_e1 = -n_yz_e1;
			n_yz_e2 = -n_yz_e2;
		}
		float d_yz_e0 = (-1.0f * sycl::dot(n_yz_e0, sycl::float2(v0.y(), v0.z()))) + sycl::max(0.0f, info.unit.y() * n_yz_e0[0]) + sycl::max(0.0f, info.unit.z() * n_yz_e0[1]);
		float d_yz_e1 = (-1.0f * sycl::dot(n_yz_e1, sycl::float2(v1.y(), v1.z()))) + sycl::max(0.0f, info.unit.y() * n_yz_e1[0]) + sycl::max(0.0f, info.unit.z() * n_yz_e1[1]);
		float d_yz_e2 = (-1.0f * sycl::dot(n_yz_e2, sycl::float2(v2.y(), v2.z()))) + sycl::max(0.0f, info.unit.y() * n_yz_e2[0]) + sycl::max(0.0f, info.unit.z() * n_yz_e2[1]);
		// ZX plane
		sycl::float2 n_zx_e0(-1.0f * e0.x(), e0.z());
		sycl::float2 n_zx_e1(-1.0f * e1.x(), e1.z());
		sycl::float2 n_zx_e2(-1.0f * e2.x(), e2.z());
		if (n.y() < 0.0f)
		{
			n_zx_e0 = -n_zx_e0;
			n_zx_e1 = -n_zx_e1;
			n_zx_e2 = -n_zx_e2;
		}
		float d_xz_e0 = (-1.0f * sycl::dot(n_zx_e0, sycl::float2(v0.z(), v0.x()))) + sycl::max(0.0f, info.unit.x() * n_zx_e0[0]) + sycl::max(0.0f, info.unit.z() * n_zx_e0[1]);
		float d_xz_e1 = (-1.0f * sycl::dot(n_zx_e1, sycl::float2(v1.z(), v1.x()))) + sycl::max(0.0f, info.unit.x() * n_zx_e1[0]) + sycl::max(0.0f, info.unit.z() * n_zx_e1[1]);
		float d_xz_e2 = (-1.0f * sycl::dot(n_zx_e2, sycl::float2(v2.z(), v2.x()))) + sycl::max(0.0f, info.unit.x() * n_zx_e2[0]) + sycl::max(0.0f, info.unit.z() * n_zx_e2[1]);

		// test possible grid boxes for overlap
		for (int z = t_bbox_grid.min.z(); z <= t_bbox_grid.max.z(); z++)
		{
			for (int y = t_bbox_grid.min.y(); y <= t_bbox_grid.max.y(); y++)
			{
				for (int x = t_bbox_grid.min.x(); x <= t_bbox_grid.max.x(); x++)
				{
					// size_t location = x + (y*info.gridsize) + (z*info.gridsize*info.gridsize);
					// if (checkBit(voxel_table, location)){ continue; }
#ifdef _DEBUG
					atomicAdd(&debug_d_n_voxels_tested, 1);
#endif
					// TRIANGLE PLANE THROUGH BOX TEST
					sycl::float3 p(x * info.unit.x(), y * info.unit.y(), z * info.unit.z());
					float nDOTp = sycl::dot(n, p);
					if ((nDOTp + d1) * (nDOTp + d2) > 0.0f)
					{
						continue;
					}

					// PROJECTION TESTS
					// XY
					sycl::float2 p_xy(p.x(), p.y());
					if ((sycl::dot(n_xy_e0, p_xy) + d_xy_e0) < 0.0f)
					{
						continue;
					}
					if ((sycl::dot(n_xy_e1, p_xy) + d_xy_e1) < 0.0f)
					{
						continue;
					}
					if ((sycl::dot(n_xy_e2, p_xy) + d_xy_e2) < 0.0f)
					{
						continue;
					}

					// YZ
					sycl::float2 p_yz(p.y(), p.z());
					if ((sycl::dot(n_yz_e0, p_yz) + d_yz_e0) < 0.0f)
					{
						continue;
					}
					if ((sycl::dot(n_yz_e1, p_yz) + d_yz_e1) < 0.0f)
					{
						continue;
					}
					if ((sycl::dot(n_yz_e2, p_yz) + d_yz_e2) < 0.0f)
					{
						continue;
					}

					// XZ
					sycl::float2 p_zx(p.z(), p.x());
					if ((sycl::dot(n_zx_e0, p_zx) + d_xz_e0) < 0.0f)
					{
						continue;
					}
					if ((sycl::dot(n_zx_e1, p_zx) + d_xz_e1) < 0.0f)
					{
						continue;
					}
					if ((sycl::dot(n_zx_e2, p_zx) + d_xz_e2) < 0.0f)
					{
						continue;
					}

#ifdef _DEBUG
					atomicAdd(&debug_d_n_voxels_marked, 1);
#endif

					if (morton_order)
					{
						size_t location =
							mortonEncode_LUT(
								x, y, z, morton256_x,
								morton256_y,
								morton256_z);
						setBit(voxel_table, location);
					}
					else
					{
						size_t location = static_cast<size_t>(x) + (static_cast<size_t>(y) * static_cast<size_t>(info.gridsize.y())) + (static_cast<size_t>(z) * static_cast<size_t>(info.gridsize.y()) * static_cast<size_t>(info.gridsize.z()));
						setBit(voxel_table, location);
					}
					continue;
				}
			}
		}
#ifdef _DEBUG
		atomicAdd(&debug_d_n_triangles, 1);
#endif
		thread_id += stride;
	}
}

void voxelize(const voxinfo &v, float *triangle_data, unsigned int *vtable, bool useThrustPath, bool morton_code)
{
	float elapsedTime;
	std::chrono::time_point<std::chrono::steady_clock> start_gpu_time, stop_gpu_time;

	// These are only used when we're not using UNIFIED memory
	unsigned int *dev_vtable; // DEVICE pointer to voxel_data
	size_t vtable_size;		  // vtable size

	// Create timers, set start time
	infra::event_ptr start_vox, stop_vox;
	std::chrono::time_point<std::chrono::steady_clock> start_vox_ct1;
	std::chrono::time_point<std::chrono::steady_clock> stop_vox_ct1;
	start_vox = new sycl::event();
	stop_vox = new sycl::event();

	// Copy morton LUT if we're encoding to morton
	if (morton_code)
	{
		start_gpu_time = std::chrono::steady_clock::now();
		sycl_device_queue
			.memcpy(morton256_x.get_ptr(), host_morton256_x,
					256 * sizeof(uint32_t))
			.wait();
		sycl_device_queue
			.memcpy(morton256_y.get_ptr(), host_morton256_y,
					256 * sizeof(uint32_t))
			.wait();
		sycl_device_queue
			.memcpy(morton256_z.get_ptr(), host_morton256_z,
					256 * sizeof(uint32_t))
			.wait();
		stop_gpu_time = std::chrono::steady_clock::now();
		total_gpu_time += std::chrono::duration<float, std::milli>(stop_gpu_time - start_gpu_time).count();
	}

	// Estimate best block and grid size using CUDA Occupancy Calculator
	int blockSize;	 // The launch configurator returned block size
	int minGridSize; // The minimum grid size needed to achieve the  maximum occupancy for a full device launch
	int gridSize;	 // The actual grid size needed, based on input size
	minGridSize = 256;
	blockSize = 16;
	// Round up according to array size
	gridSize = (v.n_triangles + blockSize - 1) / blockSize;

	if (useThrustPath)
	{ // We're not using UNIFIED memory
		vtable_size = ((size_t)v.gridsize.x() * v.gridsize.y() * v.gridsize.z()) / (size_t)8.0;
		fprintf(stdout, "[Voxel Grid] Allocating %llu kB of DEVICE memory for Voxel Grid\n", size_t(vtable_size / 1024.0f));
		start_gpu_time = std::chrono::steady_clock::now();
		dev_vtable = (unsigned int *)sycl::malloc_device(
			vtable_size, sycl_device_queue);
		sycl_device_queue
			.memset(dev_vtable, 0, vtable_size)
			.wait();
		stop_gpu_time = std::chrono::steady_clock::now();
		total_gpu_time += std::chrono::duration<float, std::milli>(stop_gpu_time - start_gpu_time).count();

		// Start voxelization
		start_vox_ct1 = std::chrono::steady_clock::now();
		*start_vox =
			sycl_device_queue.ext_oneapi_submit_barrier();
		start_gpu_time = std::chrono::steady_clock::now();
		// *stop_vox =
		sycl_device_queue.submit([&](sycl::handler &cgh)
								 {
                        morton256_x.init();
                        morton256_y.init();
                        morton256_z.init();

                        auto morton256_x_ptr_ct1 = morton256_x.get_ptr();
                        auto morton256_y_ptr_ct1 = morton256_y.get_ptr();
                        auto morton256_z_ptr_ct1 = morton256_z.get_ptr();

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, gridSize) *
                                    sycl::range<3>(1, 1, blockSize),
                                sycl::range<3>(1, 1, blockSize)),
                            [=](sycl::nd_item<3> item_ct1)
#if !defined(USE_NVIDIA_BACKEND) && !defined(USE_AMDHIP_BACKEND)
                                             [[intel::reqd_sub_group_size(32)]]
#endif
							{
                                    voxelize_triangle(v, triangle_data,
                                                      dev_vtable, morton_code,
                                                      item_ct1,
                                                      morton256_x_ptr_ct1,
                                                      morton256_y_ptr_ct1,
                                                      morton256_z_ptr_ct1);
                            }); })
			.wait();
	}
	else
	{ // UNIFIED MEMORY
		start_vox_ct1 = std::chrono::steady_clock::now();
		start_gpu_time = std::chrono::steady_clock::now();
		// *stop_vox =
		sycl_device_queue.submit([&](sycl::handler &cgh)
								 {
                            morton256_x.init();
                            morton256_y.init();
                            morton256_z.init();

                            auto morton256_x_ptr_ct1 = morton256_x.get_ptr();
                            auto morton256_y_ptr_ct1 = morton256_y.get_ptr();
                            auto morton256_z_ptr_ct1 = morton256_z.get_ptr();

                            cgh.parallel_for(
                                sycl::nd_range<3>(
                                    sycl::range<3>(1, 1, gridSize) *
                                        sycl::range<3>(1, 1, blockSize),
                                    sycl::range<3>(1, 1, blockSize)),
                                [=](sycl::nd_item<3> item_ct1)
#if !defined(USE_NVIDIA_BACKEND) && !defined(USE_AMDHIP_BACKEND)
                                             [[intel::reqd_sub_group_size(32)]]
#endif
								{
                                        voxelize_triangle(v, triangle_data,
                                                          vtable, morton_code,
                                                          item_ct1,
                                                          morton256_x_ptr_ct1,
                                                          morton256_y_ptr_ct1,
                                                          morton256_z_ptr_ct1);
                                }); })
			.wait();
	}
	// stop_vox->wait();
	stop_gpu_time = std::chrono::steady_clock::now();
	total_gpu_time += std::chrono::duration<float, std::milli>(stop_gpu_time - start_gpu_time).count();

	stop_vox_ct1 = std::chrono::steady_clock::now();
	elapsedTime = std::chrono::duration<float, std::milli>(
					  stop_vox_ct1 - start_vox_ct1)
					  .count();
	printf("Voxe[Perf] Voxelization GPU time: %.1f ms\n", elapsedTime);

	// If we're not using UNIFIED memory, copy the voxel table back and free all
	if (useThrustPath)
	{
		fprintf(stdout, "[Voxel Grid] Copying %llu kB to page-locked HOST memory\n", size_t(vtable_size / 1024.0f));
		start_gpu_time = std::chrono::steady_clock::now();
		sycl_device_queue
			.memcpy((void *)vtable, dev_vtable, vtable_size)
			.wait();
		stop_gpu_time = std::chrono::steady_clock::now();
		total_gpu_time += std::chrono::duration<float, std::milli>(stop_gpu_time - start_gpu_time).count();

		fprintf(stdout, "[Voxel Grid] Freeing %llu kB of DEVICE memory\n", size_t(vtable_size / 1024.0f));
		sycl::free(dev_vtable, sycl_device_queue);
	}

	// Destroy timers
	infra::destroy_event(start_vox);
	infra::destroy_event(stop_vox);
}
