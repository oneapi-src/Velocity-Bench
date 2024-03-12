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

#include "hip/hip_runtime.h"
#include "util_cuda.h"

// Check if ROCm requirements are met
bool initCuda()
{

	int device_count = 0;
	// Check if HIP runtime calls work at all
	hipError_t t = hipGetDeviceCount(&device_count);
	if (t != hipSuccess)
	{
		fprintf(stderr, "[ROCm] First call to HIP Runtime API failed. Are the drivers installed? \n");
		return false;
	}

	// Is there an AMD device at all?
	checkHipErrors(hipGetDeviceCount(&device_count));
	if (device_count < 1)
	{
		fprintf(stderr, "[ROCM] No AMD devices found. Make sure CUDA device is powered, connected and available. \n \n");
		fprintf(stderr, "[ROCM] On laptops: disable powersave/battery mode. \n");
		fprintf(stderr, "[ROCM] Exiting... \n");
		return false;
	}

	fprintf(stderr, "[ROCM] AMD device(s) found, picking best one \n");
	fprintf(stdout, "[ROCM] \n");
	// We have at least 1 AMD device.

	// find out which device is the best
	int best_device = -1;
	int best_compute_capability = 0;
	hipDeviceProp_t deviceProp;
	for (int device = 0; device < device_count; ++device)
	{
		hipGetDeviceProperties(&deviceProp, device);
		// choose best device
		int compute_capability = deviceProp.major * 10 + deviceProp.minor;
		if (compute_capability > best_compute_capability)
		{
			best_compute_capability = compute_capability;
			best_device = device;
		}
	}

	checkHipErrors(hipGetDeviceProperties(&deviceProp, best_device));
	fprintf(stdout, "[ROCM] Best AMD device: %s \n", deviceProp.name);
	return true;
}