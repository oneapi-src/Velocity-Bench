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

#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>

#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)

//Runtime API error
static void assert_cuda_(cudaError_t t, const char * msg)
{
    if (t == cudaSuccess)
        return;
    std::string w(msg);
    w += ": ";
    w += cudaGetErrorString(t);
    throw std::runtime_error(w);
}

//Driver API error
static void assert_cuda_(CUresult t, const char * msg)
{
    if (t == CUDA_SUCCESS)
        return;
    std::string w(msg);
    w += ": ";
    const char * str;
    if (cuGetErrorString(t, &str) == CUDA_SUCCESS)
        w += str;
    else
        w += "Unknown error";
    throw std::runtime_error(w);
}

static void assert_cufft_(cufftResult_t t, const char * msg)
{
    if (t == CUFFT_SUCCESS)
        return;
    std::string w(msg);
    w += ": ";
    switch (t)
    {
        case CUFFT_INVALID_PLAN:
            w += "CUFFT was passed an invalid plan handle"; break;
        case CUFFT_ALLOC_FAILED:
            w += "CUFFT failed to allocate GPU or CPU memory"; break;
        case CUFFT_INVALID_TYPE:
            w += "Unused"; break;
        case CUFFT_INVALID_VALUE:
            w += "User specified an invalid pointer or parameter"; break;
        case CUFFT_INTERNAL_ERROR:
            w += "Used for all driver and internal CUFFT library errors"; break;
        case CUFFT_EXEC_FAILED:
            w += "CUFFT failed to execute an FFT on the GPU"; break;
        case CUFFT_SETUP_FAILED:
            w += "The CUFFT library failed to initialize"; break;
        case CUFFT_INVALID_SIZE:
            w += "User specified an invalid transform size"; break;
        default:
            w += "Unknown CUFFT error";
    }
    throw std::runtime_error(w);
}

static void assert_cublas_(cublasStatus_t t, const char * msg)
{
    if (t == CUBLAS_STATUS_SUCCESS)
        return;
    std::string w(msg);
    w += ": ";
    switch (t)
    {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            w += "The library was not initialized"; break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            w += "The resource allocation failed"; break;
        case CUBLAS_STATUS_INVALID_VALUE:
            w += "An invalid numerical value was used as an argument"; break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            w += "An absent device architectural feature is required"; break;
        case CUBLAS_STATUS_MAPPING_ERROR:
            w += "An access to GPU memory space failed"; break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            w += "The GPU program failed to execute"; break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            w += "An internal operation failed"; break;
        default:
            w += "Unknown CUBLAS error";
    }
    throw std::runtime_error(w);
}

#ifdef _MSC_VER
#define assert_cuda(t) assert_cuda_(t, __FILE__ ":" STRINGIZE(__LINE__) " in function " __FUNCTION__)
#define assert_cufft(t) assert_cufft_(t, __FILE__ ":" STRINGIZE(__LINE__) " in function " __FUNCTION__)
#define assert_cublas(t) assert_cublas_(t, __FILE__ ":" STRINGIZE(__LINE__) " in function " __FUNCTION__)
#else
#define assert_cuda(t) assert_cuda_(t, __FILE__ ":" STRINGIZE(__LINE__))
#define assert_cufft(t) assert_cufft_(t, __FILE__ ":" STRINGIZE(__LINE__))
#define assert_cublas(t) assert_cublas_(t, __FILE__ ":" STRINGIZE(__LINE__))
#endif
