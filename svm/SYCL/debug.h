/*
MIT License

Copyright (c) 2015 University of West Bohemia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
MIT License

Modifications Copyright (C) 2023 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

SPDX-License-Identifier: MIT License
*/

#pragma once

#include <string>
#include <iostream>
#include <cuda_runtime.h>

#ifdef USE_TIMERS
#define TIME_KERNEL(kernel) \
do { \
    cudaEvent_t e1, e2; \
    cudaEventCreate(&e1); \
    cudaEventCreate(&e2); \
    cudaEventRecord(e1); \
    kernel; \
    cudaEventRecord(e2); \
    cudaEventSynchronize(e2); \
    float t; \
    cudaEventElapsedTime(&t, e1, e2); \
    std::cout << "Kernel " << #kernel << " elapsed time: " << t << " ms\n"; \
    cudaEventDestroy(e1); \
    cudaEventDestroy(e2); \
} while (false)

#define ACCUMULATE_KERNEL_TIME(timer, counter, kernel) \
do { \
    cudaEvent_t e1, e2; \
    cudaEventCreate(&e1); \
    cudaEventCreate(&e2); \
    cudaEventRecord(e1); \
    kernel; \
    cudaEventRecord(e2); \
    cudaEventSynchronize(e2); \
    float t; \
    cudaEventElapsedTime(&t, e1, e2); \
    timer += t; \
	counter++; \
    cudaEventDestroy(e1); \
    cudaEventDestroy(e2); \
} while (false)

#define PRINT_KERNEL_TIME(label, timer, counter) std::cout << "Kernel " << label << ": " << counter << " calls, average: " << ((counter > 0) ? timer/counter : 0) << " ms, total: " << timer << " ms" << std::endl;
#else
#define TIME_KERNEL(kernel) kernel
#define ACCUMULATE_KERNEL_TIME(timer, counter, kernel) kernel
#define PRINT_KERNEL_TIME(label, timer, counter)
#endif

void export_c_buffer(const void * data, size_t width, size_t height, size_t elemsize, std::string filename);
void export_cuda_buffer(const void * d_data, size_t width, size_t height, size_t elemsize, std::string filename);
void import_cuda_buffer(void * d_data, size_t width, size_t height, size_t elemsize, std::string filename);

template<typename T>
void print_cuda_buffer(const T * d_data, size_t width, size_t height, std::ostream & os = std::cout)
{
	size_t datasize = width * height;
	T * data = new T[datasize];
	assert_cuda(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

	for (int i = 0; i < width * height; i++)
		os << data[i] << " ";
	os << std::endl;

	delete[] data;
}

template<typename T>
void print_c_buffer(const T * data, size_t width, size_t height, std::ostream & os = std::cout)
{
	for (int i = 0; i < width * height; i++)
		os << data[i] << " ";
	os << std::endl;
}
