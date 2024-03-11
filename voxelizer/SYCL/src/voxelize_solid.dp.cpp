/*
        Solid voxelization based on the Schwarz-Seidel paper.
*/

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
#include <chrono>
#include "infra/infra.hpp"
#include "voxelize.dp.hpp"

// Morton LUTs for when we need them
infra::constant_memory<uint32_t, 1> morton256_x(256);
infra::constant_memory<uint32_t, 1> morton256_y(256);
infra::constant_memory<uint32_t, 1> morton256_z(256);

#ifdef _DEBUG
__device__ size_t debug_d_n_voxels_marked = 0;
__device__ size_t debug_d_n_triangles = 0;
__device__ size_t debug_d_n_voxels_tested = 0;
#endif

#define float_error 0.000001

// use Xor for voxels whose corresponding bits have to flipped
__inline__ void setBitXor(unsigned int *voxel_table, size_t index)
{
        size_t int_location = index / size_t(32);
        unsigned int bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
        unsigned int mask = 1 << bit_pos;
        // atomicXor(&(voxel_table[int_location]), mask);
        infra::atomic_fetch_xor<unsigned int, sycl::access::address_space::generic_space>(&(voxel_table[int_location]), mask);
}

// check the location with point and triangle
inline int check_point_triangle(sycl::float2 v0, sycl::float2 v1, sycl::float2 v2, sycl::float2 point)
{
        sycl::float2 PA = point - v0;
        sycl::float2 PB = point - v1;
        sycl::float2 PC = point - v2;

        float t1 = PA.x() * PB.y() - PA.y() * PB.x();
        if (sycl::fabs(t1) < float_error && PA.x() * PB.x() <= 0 && PA.y() * PB.y() <= 0)
                return 1;

        float t2 = PB.x() * PC.y() - PB.y() * PC.x();
        if (sycl::fabs(t2) < float_error && PB.x() * PC.x() <= 0 && PB.y() * PC.y() <= 0)
                return 2;

        float t3 = PC.x() * PA.y() - PC.y() * PA.x();
        if (sycl::fabs(t3) < float_error && PC.x() * PA.x() <= 0 && PC.y() * PA.y() <= 0)
                return 3;

        if (t1 * t2 > 0 && t1 * t3 > 0)
                return 0;
        else
                return -1;
}

// find the x coordinate of the voxel
inline float get_x_coordinate(sycl::float3 n, sycl::float3 v0, sycl::float2 point)
{
        return (-(n.y() * (point.x() - v0.y()) + n.z() * (point.y() - v0.z())) / n.x() + v0.x());
}

// check the triangle is counterclockwise or not
inline bool checkCCW(sycl::float2 v0, sycl::float2 v1, sycl::float2 v2)
{
        sycl::float2 e0 = v1 - v0;
        sycl::float2 e1 = v2 - v0;
        float result = e0.x() * e1.y() - e1.x() * e0.y();
        if (result > 0)
                return true;
        else
                return false;
}

// top-left rule
inline bool TopLeftEdge(sycl::float2 v0, sycl::float2 v1)
{
        return ((v1.y() < v0.y()) || (v1.y() == v0.y() && v0.x() > v1.x()));
}

// generate solid voxelization
void voxelize_triangle_solid(voxinfo info, float *triangle_data, unsigned int *voxel_table, bool morton_order,
                             sycl::nd_item<3> item_ct1, uint32_t *morton256_x,
                             uint32_t *morton256_y, uint32_t *morton256_z)
{
        size_t thread_id = item_ct1.get_local_id(2) +
                           item_ct1.get_group(2) * item_ct1.get_local_range(2);
        size_t stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

        while (thread_id < info.n_triangles)
        {                                 // every thread works on specific triangles in its stride
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
                if (sycl::fabs(n.x()) < float_error)
                        return;

                // Calculate the projection of three point into yoz plane
                sycl::float2 v0_yz = sycl::float2(v0.y(), v0.z());
                sycl::float2 v1_yz = sycl::float2(v1.y(), v1.z());
                sycl::float2 v2_yz = sycl::float2(v2.y(), v2.z());

                // set the triangle counterclockwise
                if (!checkCCW(v0_yz, v1_yz, v2_yz))
                {
                        sycl::float2 v3 = v1_yz;
                        v1_yz = v2_yz;
                        v2_yz = v3;
                }

                // COMPUTE TRIANGLE BBOX IN GRID
                // Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
                sycl::float2 bbox_max = sycl::max(v0_yz, sycl::max(v1_yz, v2_yz));
                sycl::float2 bbox_min = sycl::min(v0_yz, sycl::min(v1_yz, v2_yz));

                sycl::float2 bbox_max_grid =
                    sycl::float2(sycl::floor(bbox_max.x() / info.unit.y() - 0.5),
                                 sycl::floor(bbox_max.y() / info.unit.z() - 0.5));
                sycl::float2 bbox_min_grid =
                    sycl::float2(sycl::ceil(bbox_min.x() / info.unit.y() - 0.5),
                                 sycl::ceil(bbox_min.y() / info.unit.z() - 0.5));

                for (int y = bbox_min_grid.x(); y <= bbox_max_grid.x(); y++)
                {
                        for (int z = bbox_min_grid.y(); z <= bbox_max_grid.y(); z++)
                        {
                                sycl::float2 point = sycl::float2((y + 0.5) * info.unit.y(), (z + 0.5) * info.unit.z());
                                int checknum = check_point_triangle(v0_yz, v1_yz, v2_yz, point);
                                if ((checknum == 1 && TopLeftEdge(v0_yz, v1_yz)) || (checknum == 2 && TopLeftEdge(v1_yz, v2_yz)) || (checknum == 3 && TopLeftEdge(v2_yz, v0_yz)) || (checknum == 0))
                                {
                                        int xmax = int(get_x_coordinate(n, v0, point) / info.unit.x() - 0.5);
                                        for (int x = 0; x <= xmax; x++)
                                        {
                                                if (morton_order)
                                                {
                                                        size_t location =
                                                            mortonEncode_LUT(
                                                                x, y, z,
                                                                morton256_x,
                                                                morton256_y,
                                                                morton256_z);
                                                        setBitXor(voxel_table, location);
                                                }
                                                else
                                                {
                                                        size_t location = static_cast<size_t>(x) + (static_cast<size_t>(y) * static_cast<size_t>(info.gridsize.y())) + (static_cast<size_t>(z) * static_cast<size_t>(info.gridsize.y()) * static_cast<size_t>(info.gridsize.z()));
                                                        setBitXor(voxel_table, location);
                                                }
                                                continue;
                                        }
                                }
                        }
                }

                // sanity check: atomically count triangles
                // atomicAdd(&triangles_seen_count, 1);
                thread_id += stride;
        }
}

void voxelize_solid(const voxinfo &v, float *triangle_data, unsigned int *vtable, bool useThrustPath, bool morton_code)
{
        float elapsedTime;

        // These are only used when we're not using UNIFIED memory
        unsigned int *dev_vtable; // DEVICE pointer to voxel_data
        size_t vtable_size;       // vtable size

        // Create timers, set start time
        infra::event_ptr start_vox, stop_vox;
        std::chrono::time_point<std::chrono::steady_clock> start_vox_ct1;
        std::chrono::time_point<std::chrono::steady_clock> stop_vox_ct1;
        std::chrono::time_point<std::chrono::steady_clock> start_gpu_time, stop_gpu_time;
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
        int blockSize;   // The launch configurator returned block size
        int minGridSize; // The minimum grid size needed to achieve the  maximum occupancy for a full device launch
        int gridSize;    // The actual grid size needed, based on input size
        minGridSize = 256;
        blockSize = 256;
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
                            [=](sycl::nd_item<3> item_ct1) {
                                    voxelize_triangle_solid(
                                        v, triangle_data, dev_vtable,
                                        morton_code, item_ct1,
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
                                [=](sycl::nd_item<3> item_ct1) {
                                        voxelize_triangle_solid(
                                            v, triangle_data, vtable,
                                            morton_code, item_ct1,
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
        printf("[Perf] Voxelization GPU time: %.1f ms\n", elapsedTime);

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
        infra::destroy_event(start_vox);
        infra::destroy_event(stop_vox);
}