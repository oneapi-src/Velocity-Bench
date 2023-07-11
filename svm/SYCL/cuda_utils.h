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

#include <cstdio>
#include <algorithm>

#define SUCCESS 0
#define FAILURE 1

#define TRUE 1
#define FALSE 0

#ifdef __CUDACC__
template<typename T>
__host__ __device__ T getgriddim(T totallen, T blockdim)
{
    return (totallen + blockdim - (T)1) / blockdim;
}

template<typename T>
__host__ __device__ T rounduptomult(T x, T m)
{
    return ((x + m - (T)1) / m) * m;
}

template<typename T>
__device__ T warpReduceSum(T val)
{
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
		val += __shfl_down(val, s);
	return val;
}

template<typename T>
__device__ static void swap_dev(T & a, T & b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

template<typename T>
__global__ static void kernelMemset(T * mem, T v, int n)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    while (k < n)
    {
        mem[k] = v;
        k += gridDim.x * blockDim.x;
    }
}

template<typename T>
static void memsetCuda(T * d_mem, T v, int n)
{
    dim3 dimBlock(256);
    dim3 dimGrid(std::min(2048, getgriddim<int>(n, dimBlock.x)));
    kernelMemset<T><<<dimGrid, dimBlock>>>(d_mem, v, n);
}

template<typename T>
__device__ void blockReduceSum(T * v)
{
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            v[threadIdx.x] += v[threadIdx.x + s];
        __syncthreads();
    }
}

template<typename TVal, typename TIdx>
__device__ static int blockMaxReduce(const TVal * v, TIdx * i)
{
    i[threadIdx.x] = threadIdx.x;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            if (v[i[threadIdx.x + s]] > v[i[threadIdx.x]] || (v[i[threadIdx.x + s]] == v[i[threadIdx.x]] && i[threadIdx.x + s] < i[threadIdx.x]))
            {
                i[threadIdx.x] = i[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    return i[0];
}

template<typename T1, typename T2>
__device__ void blockMinReduce2(T1 * v1, T2 * v2)
{
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            if (v1[threadIdx.x + s] < v1[threadIdx.x])
            {
                v1[threadIdx.x] = v1[threadIdx.x + s];
                v2[threadIdx.x] = v2[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
}

template<bool desc, typename K, typename T>
__device__ void blockBitonicSort(K * idx, T * val)
{
    //int i = blockDim.x * blockIdx.x + threadIdx.x;
    int i = threadIdx.x;
    for (int k = 2; k <= blockDim.x; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            int ixj = i^j;
            if (ixj > i)
            {
                if (bool(i & k) == desc)
                {
                    if (val[i] > val[ixj])
                    {
                        swap_dev(val[i], val[ixj]);
                        swap_dev(idx[i], idx[ixj]);
                    }
                }
                else
                {
                    if (val[i] < val[ixj])
                    {
                        swap_dev(val[i], val[ixj]);
                        swap_dev(idx[i], idx[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

template<bool desc, typename K, typename T>
__device__ void blockBitonicSortN(K * idx, T * val, int n)
{
    //int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int k = 2; k <= n; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            for (int i = threadIdx.x; i < n; i += blockDim.x)
            {
                int ixj = i^j;
                if (ixj > i)
                {
                    if (bool(i & k) == desc)
                    {
                        if (val[i] > val[ixj])
                        {
                            swap_dev(val[i], val[ixj]);
                            swap_dev(idx[i], idx[ixj]);
                        }
                    }
                    else
                    {
                        if (val[i] < val[ixj])
                        {
                            swap_dev(val[i], val[ixj]);
                            swap_dev(idx[i], idx[ixj]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

//inter-block synchronization stuff
#define DEFINE_SYNC_BUFFERS(num) __device__ int d_sync_buffer[num]
#define SYNC_BUFFER_DEF int sync_buffer_id
#define SYNC_RESET(id) \
do { \
    int * p; \
    assert_cuda(cudaGetSymbolAddress((void * *)&p, d_sync_buffer)); \
    assert_cuda(cudaMemset(p + id, 0, sizeof(int))); \
} while(false)
#define SYNC_BUFFER(id) id

//global synchronization, last block continue, others return
#define WAIT_FOR_THE_FINAL_BLOCK \
do { \
	__threadfence(); \
	__shared__ int value; \
	if (threadIdx.x + threadIdx.y == 0) value = 1 + atomicAdd(d_sync_buffer + sync_buffer_id, 1); \
	__syncthreads(); \
	if (value < gridDim.z * gridDim.y * gridDim.x) return; \
    if (threadIdx.x + threadIdx.y == 0) d_sync_buffer[sync_buffer_id] = 0; \
} while (false)

#endif

#define CUDA_SAFE_MALLOC(MEM_PTR, SIZE) CUDA_SAFE_MFREE(*MEM_PTR) \
if (cudaMalloc(MEM_PTR, SIZE) != cudaSuccess) { \
	printf("Error(%s): Unable to allocate %d B of memory on GPU!\n", cudaGetErrorString(cudaGetLastError()), SIZE); \
	exit(EXIT_FAILURE); \
}

#define CUDA_SAFE_MALLOC_2D(MEM_PTR, PITCH, WIDTH, HEIGHT) CUDA_SAFE_MFREE(*MEM_PTR) \
if (cudaMallocPitch(MEM_PTR, PITCH, WIDTH, HEIGHT) != cudaSuccess) { \
	printf("Error(%s): Unable to allocate %d B of 2D memory on GPU!\n", cudaGetErrorString(cudaGetLastError()), WIDTH * HEIGHT); \
	exit(EXIT_FAILURE); \
}

#define CUDA_SAFE_MEMCPY(DST, SRC, SIZE, KIND) \
if (cudaMemcpy(DST, SRC, SIZE, KIND) != cudaSuccess) { \
	printf("Error: Unable to copy %d B of memory from RAM (%llX) to GPU (%llX)!\n", SIZE, SRC, DST); \
	exit(EXIT_FAILURE); \
}

#define CUDA_SAFE_MEMCPY_2D(DST, SRC, PITCH, WIDTH, HEIGHT, KIND) \
if (cudaMemcpy2D(DST, PITCH, SRC, WIDTH, WIDTH, HEIGHT, KIND) != cudaSuccess) { \
	printf("Error: Unable to copy %d B of 2D memory from RAM (%llX) to GPU (%llX) due to %s !\n", WIDTH * HEIGHT, SRC, DST, cudaGetErrorString(cudaGetLastError())); \
	exit(EXIT_FAILURE); \
}

#define CUDA_SAFE_MFREE(MEM_PTR) if (MEM_PTR != NULL) { \
if (cudaFree(MEM_PTR) != cudaSuccess) { \
	printf("Error(%s): Unable to free GPU memory at address %llX!\n", cudaGetErrorString(cudaGetLastError()), MEM_PTR); \
	exit(EXIT_FAILURE); \
} \
MEM_PTR = NULL; \
}

#define CUDA_SAFE_MFREE_HOST(MEM_PTR) if (MEM_PTR != NULL) { \
if (cudaFreeHost(MEM_PTR) != cudaSuccess) { \
	printf("Error(%s): Unable to free host data at address %llX!\n", cudaGetErrorString(cudaGetLastError()), MEM_PTR); \
	exit(EXIT_FAILURE); \
} \
MEM_PTR = NULL; \
}
