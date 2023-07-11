/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @brief Utility/Error Checking code.
 *
 * @file cuda_utils.cu
 * @date 2018-04-04
 * Copyright (C) 2012-2017 Orange Owl Solutions.
 */


 /*
 CUDA Utilities - Utilities for high performance CPUs/C/C++ and GPUs/CUDA computing library.

    Copyright (C) 2012-2017 Orange Owl Solutions.


    CUDA Utilities is free software: you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDA Utilities is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Lesser GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Bluebird Library.  If not, see <http://www.gnu.org/licenses/>.


    For any request, question or bug reporting please visit http://www.orangeowlsolutions.com/
    or send an e-mail to: info@orangeowlsolutions.com
*/

#include "include/utils/cuda_utils.h"

/*******************/
/* iDivUp FUNCTION */
/*******************/
//extern "C" int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
__host__ __device__ int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(hipError_t code, const char *file, int line, bool abort = true)
{
	if (code != hipSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
		if (abort) { exit(code); }
	}
}

extern "C" void GpuErrorCheck(hipError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

/*************************/
/* CUBLAS ERROR CHECKING */
/*************************/
static const char *_cublasGetErrorEnum(hipblasStatus_t error)
{
	switch (error)
	{
	case HIPBLAS_STATUS_SUCCESS:
		return "HIPBLAS_STATUS_SUCCESS";

	case HIPBLAS_STATUS_NOT_INITIALIZED:
		return "HIPBLAS_STATUS_NOT_INITIALIZED";

	case HIPBLAS_STATUS_ALLOC_FAILED:
		return "HIPBLAS_STATUS_ALLOC_FAILED";

	case HIPBLAS_STATUS_INVALID_VALUE:
		return "HIPBLAS_STATUS_INVALID_VALUE";

	case HIPBLAS_STATUS_ARCH_MISMATCH:
		return "HIPBLAS_STATUS_ARCH_MISMATCH";

	case HIPBLAS_STATUS_MAPPING_ERROR:
		return "HIPBLAS_STATUS_MAPPING_ERROR";

	case HIPBLAS_STATUS_EXECUTION_FAILED:
		return "HIPBLAS_STATUS_EXECUTION_FAILED";

	case HIPBLAS_STATUS_INTERNAL_ERROR:
		return "HIPBLAS_STATUS_INTERNAL_ERROR";

	case HIPBLAS_STATUS_NOT_SUPPORTED:
		return "HIPBLAS_STATUS_NOT_SUPPORTED";

	case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
	    return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    
    case HIPBLAS_STATUS_INVALID_ENUM:
        return "HIPBLAS_STATUS_INVALID_ENUM";
    
    case HIPBLAS_STATUS_UNKNOWN:
        return "HIPBLAS_STATUS_UNKNOWN";
	}

	return "<unknown>";
}

inline void __CublasSafeCall(hipblasStatus_t err, const char *file, const int line)
{
	if (HIPBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d, error: %s\nterminating!\n", __FILE__, __LINE__, \
			_cublasGetErrorEnum(err)); \
			assert(0); \
	}
}

extern "C" void CublasSafeCall(hipblasStatus_t err) { __CublasSafeCall(err, __FILE__, __LINE__); }

/***************************/
/* CUSPARSE ERROR CHECKING */
/***************************/
static const char *_cusparseGetErrorEnum(hipsparseStatus_t error)
{
	switch (error)
	{

	case HIPSPARSE_STATUS_SUCCESS:
		return "HIPSPARSE_STATUS_SUCCESS";

	case HIPSPARSE_STATUS_NOT_INITIALIZED:
		return "HIPSPARSE_STATUS_NOT_INITIALIZED";

	case HIPSPARSE_STATUS_ALLOC_FAILED:
		return "HIPSPARSE_STATUS_ALLOC_FAILED";

	case HIPSPARSE_STATUS_INVALID_VALUE:
		return "HIPSPARSE_STATUS_INVALID_VALUE";

	case HIPSPARSE_STATUS_ARCH_MISMATCH:
		return "HIPSPARSE_STATUS_ARCH_MISMATCH";

	case HIPSPARSE_STATUS_MAPPING_ERROR:
		return "HIPSPARSE_STATUS_MAPPING_ERROR";

	case HIPSPARSE_STATUS_EXECUTION_FAILED:
		return "HIPSPARSE_STATUS_EXECUTION_FAILED";

	case HIPSPARSE_STATUS_INTERNAL_ERROR:
		return "HIPSPARSE_STATUS_INTERNAL_ERROR";

	case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

	case HIPSPARSE_STATUS_ZERO_PIVOT:
		return "HIPSPARSE_STATUS_ZERO_PIVOT";

	case HIPSPARSE_STATUS_NOT_SUPPORTED:
		return "HIPSPARSE_STATUS_NOT_SUPPORTED";

	case HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES:
		return "HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES";
	}

	return "<unknown>";
}

inline void __CusparseSafeCall(hipsparseStatus_t err, const char *file, const int line)
{
	if (HIPSPARSE_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUSPARSE error in file '%s', line %d, error %s\nterminating!\n", __FILE__, __LINE__, \
			_cusparseGetErrorEnum(err)); \
			assert(0); \
	}
}

extern "C" void CusparseSafeCall(hipsparseStatus_t err) { __CusparseSafeCall(err, __FILE__, __LINE__); }

/************************/
/* CUFFT ERROR CHECKING */
/************************/

static const char *_cufftGetErrorEnum(hipfftResult error)
{
    switch (error)
    {
        case HIPFFT_SUCCESS:
            return "HIPFFT_SUCCESS";

        case HIPFFT_INVALID_PLAN:
            return "HIPFFT_INVALID_PLAN";

        case HIPFFT_ALLOC_FAILED:
            return "HIPFFT_ALLOC_FAILED";

        case HIPFFT_INVALID_TYPE:
            return "HIPFFT_INVALID_TYPE";

        case HIPFFT_INVALID_VALUE:
            return "HIPFFT_INVALID_VALUE";

        case HIPFFT_INTERNAL_ERROR:
            return "HIPFFT_INTERNAL_ERROR";

        case HIPFFT_EXEC_FAILED:
            return "HIPFFT_EXEC_FAILED";

        case HIPFFT_SETUP_FAILED:
            return "HIPFFT_SETUP_FAILED";

        case HIPFFT_INVALID_SIZE:
            return "HIPFFT_INVALID_SIZE";

        case HIPFFT_UNALIGNED_DATA:
            return "HIPFFT_UNALIGNED_DATA";

        case HIPFFT_INVALID_DEVICE:
            return "HIPFFT_INVALID_DEVICE";

        case HIPFFT_INCOMPLETE_PARAMETER_LIST:
            return "HIPFFT_INCOMPLETE_PARAMETER_LIST";

        case HIPFFT_PARSE_ERROR:
            return "HIPFFT_PARSE_ERROR";

        case HIPFFT_NO_WORKSPACE:
            return "HIPFFT_NO_WORKSPACE";

        case HIPFFT_NOT_SUPPORTED:
            return "HIPFFT_NOT_SUPPORTED";

        case HIPFFT_NOT_IMPLEMENTED:
            return "HIPFFT_NOT_IMPLEMENTED";

        default:
            return "<unknown>";
    }
}

inline void __cufftSafeCall(hipfftResult err, const char *file, const int line)
{
    if( HIPFFT_SUCCESS != err) {
		fprintf(stderr,
                "CUFFT error in file '%s', line %d, error %s\nterminating!\n",
                __FILE__,
                __LINE__,
				_cufftGetErrorEnum(err));
		GpuErrorCheck(hipDeviceReset());
        assert(0);
    }
}

extern "C" void CufftSafeCall(hipfftResult err) { __cufftSafeCall(err, __FILE__, __LINE__); }


/// END OF CUDA UTILITIES
