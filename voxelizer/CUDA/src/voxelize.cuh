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

#pragma once

// Commun functions for both the solid and non-solid voxelization methods

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define GLM_FORCE_CUDA
// #define GLM_FORCE_PURE (not needed anymore with recent GLM versions)
#include <glm/glm.hpp>

#include <iostream>
#include "util.h"
#include "util_cuda.h"
#include "morton_LUTs.h"
#include <chrono>

extern float total_gpu_time;

// Morton LUTs for when we need them
__constant__ uint32_t morton256_x[256];
__constant__ uint32_t morton256_y[256];
__constant__ uint32_t morton256_z[256];

// Encode morton code using LUT table
__device__ inline uint64_t mortonEncode_LUT(unsigned int x, unsigned int y, unsigned int z)
{
	uint64_t answer = 0;
	answer = morton256_z[(z >> 16) & 0xFF] |
			 morton256_y[(y >> 16) & 0xFF] |
			 morton256_x[(x >> 16) & 0xFF];
	answer = answer << 48 |
			 morton256_z[(z >> 8) & 0xFF] |
			 morton256_y[(y >> 8) & 0xFF] |
			 morton256_x[(x >> 8) & 0xFF];
	answer = answer << 24 |
			 morton256_z[(z) & 0xFF] |
			 morton256_y[(y) & 0xFF] |
			 morton256_x[(x) & 0xFF];
	return answer;
}