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

#include <chrono>
#include <iostream>
#include "util_cuda.h"

// Check if CUDA requirements are met
bool initCuda()
{

	int device_count = 0;
	// Check if CUDA runtime calls work at all
	cudaError t = cudaGetDeviceCount(&device_count);
	if (t != cudaSuccess)
	{
		fprintf(stderr, "[CUDA] First call to CUDA Runtime API failed. Are the drivers installed? \n");
		return false;
	}

	// Is there a CUDA device at all?
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	if (device_count < 1)
	{
		fprintf(stderr, "[CUDA] No CUDA devices found. Make sure CUDA device is powered, connected and available. \n \n");
		fprintf(stderr, "[CUDA] On laptops: disable powersave/battery mode. \n");
		fprintf(stderr, "[CUDA] Exiting... \n");
		return false;
	}

	fprintf(stderr, "[CUDA] CUDA device(s) found \n");
	fprintf(stdout, "[CUDA] ");
	// We have at least 1 CUDA device, so now select the fastest (method from Nvidia helper library)
	int device = findCudaDevice(0, 0);
	// int device = 0; // setting device to a default value 0.

	// Print available device memory
	cudaDeviceProp properties;
	checkCudaErrors(cudaGetDeviceProperties(&properties, device));
	fprintf(stdout, "[CUDA] Best device: %s \n", properties.name);

	// Check compute capability
	if (properties.major < 2)
	{
		fprintf(stderr, "[CUDA] Your cuda device has compute capability %i.%i. We need at least 2.0 for atomic operations. \n", properties.major, properties.minor);
		return false;
	}
	return true;
}