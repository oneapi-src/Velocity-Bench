/* Copyright (C) 2023 Intel Corporation
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 * SPDX-License-Identifier: MIT
 */

#ifndef DL_CIFAR_CUBLAS_ROUTINES_H_
#define DL_CIFAR_CUBLAS_ROUTINES_H_

#include <iostream>
#include <exception>
#include <iostream>
#include <string>
#include "tracing.h"
#include "handle.h"

#include <sycl.hpp>


#if defined(USE_CUBLAS)
//#include <sycl/backend/cuda.hpp>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "error_handling.h"
#elif defined(USE_ROCBLAS)
#include "hip/hip_runtime_api.h"
#include <rocblas.h>
#include "error_handling.h"
#else
#include "oneapi/dnnl/dnnl.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "dnnl_sycl.hpp"
#endif
#include <oneapi/mkl.hpp>

using oneapi::mkl::blas::gemm;
using oneapi::mkl::transpose;
//using oneapi::mkl::blas::row_major;
#endif

using namespace dl_cifar::common;

class BlasRoutines {
    public:
#ifdef USE_ROCBLAS
        static rocblas_status_ doAxpy(LangHandle *langHandle, int noOfElems, float *scalingFactor, float *d_x, float *d_y) {
            Tracer::func_begin("MklRoutines::doAxpy");    

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    //cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    //cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

                    //auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    //cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    constexpr int INCX = 1;
                    assertBlasInvar(rocblas_saxpy(*(langHandle->getRocblasHandle()), noOfElems, scalingFactor, d_x, 1, d_y, 1));
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDlApiInvar(hipDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();


            Tracer::func_end("MklRoutines::doAxpy");  
            return rocblas_status_success;
        }

        static rocblas_status_ scaleVector(LangHandle *langHandle, int noOfElements, float *scalingFactor, float* vector, int inc) {
            Tracer::func_begin("MklRoutines::scaleVector");    

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    //cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    //cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

                    // auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    // cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    assertBlasInvar(rocblas_sscal(*(langHandle->getRocblasHandle()), noOfElements, scalingFactor, vector, inc));
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDlApiInvar(hipDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();

            //cublasStatus_t status = cublasSscal(*(langHandle->getCublasHandle()), noOfElements, scalingFactor, vector, inc);
            Tracer::func_end("MklRoutines::scaleVector");  
            return rocblas_status_success;
        }
        
        static rocblas_status_ doMatMul(LangHandle *langHandle, int m, int k, int n, const float *A, const float *B, float *C) {
            Tracer::func_begin("MklRoutines::doMatMul");    

            float alpha_ = 1.0;
            float beta_ = 0.0;

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    //cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    //cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());
                    
                    // auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    // cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    assertBlasInvar(rocblas_sgemm(*(langHandle->getRocblasHandle()), rocblas_operation_none, rocblas_operation_none,
                                                n, m, k, &alpha_, B, n, A, k, &beta_, C, n));
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDlApiInvar(hipDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();

            //cublasStatus_t status =  cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_N,
            //                                        n, m, k, &alpha_, B, n, A, k, &beta_, C, n);
            Tracer::func_end("MklRoutines::doMatMul");  
            return rocblas_status_success;                                        
        }

    

        static rocblas_status_ doMatMulTraA(LangHandle *langHandle, int m, int k, int n, float *A, const float *B, float *C) {
            Tracer::func_begin("MklRoutines::doMatMulTraA");    

            float alpha_ = 1.0;
            float beta_ = 0.0;

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    //cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    //cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

                    // auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    // cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    assertBlasInvar(rocblas_sgemm(*(langHandle->getRocblasHandle()), rocblas_operation_none, rocblas_operation_transpose,
                                                n, m, k, &alpha_, B, n, A, m, &beta_, C, n));
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDevApiInvar(hipDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();

            //cublasStatus_t status =  cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_T,
            //                                        n, m, k, &alpha_, B, n, A, m, &beta_, C, n);
            Tracer::func_end("MklRoutines::doMatMulTraA");  
            return rocblas_status_success;                                        
        }


        static rocblas_status_ doMatMulTraB(LangHandle *langHandle, int m, int k, int n, float *A, const float *B, float *C) {
            Tracer::func_begin("MklRoutines::doMatMulTraB");    

            float alpha_ = 1.0;
            float beta_ = 0.0;

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    //cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    //cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

                    // auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    // cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    assertBlasInvar(rocblas_sgemm(*(langHandle->getRocblasHandle()), rocblas_operation_transpose, rocblas_operation_none,
                                                n, m, k, &alpha_, B, k, A, k, &beta_, C, n));
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDevApiInvar(hipDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();

            //cublasStatus_t status =  cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_T, CUBLAS_OP_N,
            //                                        n, m, k, &alpha_, B, k, A, k, &beta_, C, n);
            Tracer::func_end("MklRoutines::doMatMulTraB");  
            return rocblas_status_success;                                        
        }

    
        static rocblas_status_ doBatchedMatMul(LangHandle *langHandle, int m, int k, int n, 
                                                const float *A[], const float *B[], float *C[], int batchCount) {
            Tracer::func_begin("MklRoutines::doBatchedMatMul");    

            float alpha_ = 1.0;
            float beta_ = 0.0;

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    //cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    //cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

                    // auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    // cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    assertBlasInvar(rocblas_sgemm_batched(*(langHandle->getRocblasHandle()), rocblas_operation_none, rocblas_operation_none,
                                                n, m, k, &alpha_, B, n, A, k, &beta_, C, n, batchCount));
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDevApiInvar(hipDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();

            //cublasStatus_t status =  cublasSgemmBatched(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_N,
            //                                        n, m, k, &alpha_, B, n, A, k, &beta_, C, n, batchCount);
            Tracer::func_end("MklRoutines::doBatchedMatMul");  
            return rocblas_status_success;                                        
        }    

#elif defined(USE_CUBLAS)
        static cublasStatus_t doAxpy(LangHandle *langHandle, int noOfElems, float *scalingFactor, float *d_x, float *d_y) {
            Tracer::func_begin("MklRoutines::doAxpy");    

            //cublasStatus_t status = cublasSaxpy(*(langHandle->getCublasHandle()), noOfElems, scalingFactor, d_x, 1, d_y, 1);

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

                    //auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    //cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    constexpr int INCX = 1;
                    assertBlasInvar(cublasSaxpy(*(langHandle->getCublasHandle()), noOfElems, scalingFactor, d_x, 1, d_y, INCX));
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDevApiInvar(cudaDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();


            Tracer::func_end("MklRoutines::doAxpy");  
            return CUBLAS_STATUS_SUCCESS;
        }

        static cublasStatus_t scaleVector(LangHandle *langHandle, int noOfElements, float *scalingFactor, float* vector, int inc) {
            Tracer::func_begin("MklRoutines::scaleVector");    

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

                    // auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    // cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    cublasSscal(*(langHandle->getCublasHandle()), noOfElements, scalingFactor, vector, inc);
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDevApiInvar(cudaDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();

            //cublasStatus_t status = cublasSscal(*(langHandle->getCublasHandle()), noOfElements, scalingFactor, vector, inc);
            Tracer::func_end("MklRoutines::scaleVector");  
            return CUBLAS_STATUS_SUCCESS;
        }
        
        static cublasStatus_t doMatMul(LangHandle *langHandle, int m, int k, int n, const float *A, const float *B, float *C) {
            Tracer::func_begin("MklRoutines::doMatMul");    

            float alpha_ = 1.0;
            float beta_ = 0.0;

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());
                    
                    // auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    // cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_N,
                                                    n, m, k, &alpha_, B, n, A, k, &beta_, C, n);
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDevApiInvar(cudaDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();

            //cublasStatus_t status =  cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_N,
            //                                        n, m, k, &alpha_, B, n, A, k, &beta_, C, n);
            Tracer::func_end("MklRoutines::doMatMul");  
            return CUBLAS_STATUS_SUCCESS;                                        
        }

    

        static cublasStatus_t doMatMulTraA(LangHandle *langHandle, int m, int k, int n, float *A, const float *B, float *C) {
            Tracer::func_begin("MklRoutines::doMatMulTraA");    

            float alpha_ = 1.0;
            float beta_ = 0.0;

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

                    // auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    // cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_T,
                                                    n, m, k, &alpha_, B, n, A, m, &beta_, C, n);
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDevApiInvar(cudaDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();

            //cublasStatus_t status =  cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_T,
            //                                        n, m, k, &alpha_, B, n, A, m, &beta_, C, n);
            Tracer::func_end("MklRoutines::doMatMulTraA");  
            return CUBLAS_STATUS_SUCCESS;                                        
        }


        static cublasStatus_t doMatMulTraB(LangHandle *langHandle, int m, int k, int n, float *A, const float *B, float *C) {
            Tracer::func_begin("MklRoutines::doMatMulTraB");    

            float alpha_ = 1.0;
            float beta_ = 0.0;

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

                    // auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    // cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_T, CUBLAS_OP_N,
                                                    n, m, k, &alpha_, B, k, A, k, &beta_, C, n);
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDevApiInvar(cudaDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();

            //cublasStatus_t status =  cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_T, CUBLAS_OP_N,
            //                                        n, m, k, &alpha_, B, k, A, k, &beta_, C, n);
            Tracer::func_end("MklRoutines::doMatMulTraB");  
            return CUBLAS_STATUS_SUCCESS;                                        
        }

    
        static cublasStatus_t doBatchedMatMul(LangHandle *langHandle, int m, int k, int n, 
                                                const float *A[], const float *B[], float *C[], int batchCount) {
            Tracer::func_begin("MklRoutines::doBatchedMatMul");    

            float alpha_ = 1.0;
            float beta_ = 0.0;

            langHandle->getSyclQueue()->submit([&](sycl::handler &cgh) {
                //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                    cublasSetStream(*(langHandle->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

                    // auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
                    // cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
                    //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
                    cublasSgemmBatched(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_N,
                                                    n, m, k, &alpha_, B, n, A, k, &beta_, C, n, batchCount);
                    //cublasDestroy(handle);
                    //cudaStreamSynchronize(cudaStreamHandle);
                    assertDevApiInvar(cudaDeviceSynchronize());
                });
            });
            langHandle->getSyclQueue()->wait_and_throw();

            //cublasStatus_t status =  cublasSgemmBatched(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_N,
            //                                        n, m, k, &alpha_, B, n, A, k, &beta_, C, n, batchCount);
            Tracer::func_end("MklRoutines::doBatchedMatMul");  
            return CUBLAS_STATUS_SUCCESS;                                        
        }    

#else
        static int doAxpy(LangHandle *langHandle, int noOfElems, float *alpha, float *d_x, float *d_y) {
            Tracer::func_begin("BlasRoutines::doAxpy");    
            try {
                oneapi::mkl::blas::row_major::axpy(*(langHandle->getSyclQueue()), noOfElems, *alpha, d_x, 1, d_y, 1);
            }
            catch (sycl::exception const& e) {
                std::cout << "\t\tCaught synchronous SYCL exception during axpy:\n"
                    << e.what() << std::endl;
                    return 1;
            }
            catch (std::exception const& e) {
                std::cout << "\t\tCaught synchronous STL exception during axpy:\n"
                    << e.what() << std::endl;
                    return 2;
            }

            langHandle->getSyclQueue()->wait_and_throw();
            Tracer::func_end("BlasRoutines::doAxpy");     
            return 0; 
        }

        static int scaleVector(LangHandle *langHandle, int noOfElems, float *alpha, float *d_x, int inc) {
            Tracer::func_begin("BlasRoutines::scaleVector");
            try {
                oneapi::mkl::blas::row_major::scal(*(langHandle->getSyclQueue()), noOfElems, *alpha, d_x, inc);
            }
            catch (sycl::exception const& e) {
                std::cout << "\t\tCaught synchronous SYCL exception during scal:\n"
                    << e.what() << std::endl;
                    return 1;
            }
            catch (std::exception const& e) {
                std::cout << "\t\tCaught synchronous STL exception during scal:\n"
                    << e.what() << std::endl;
                    return 2;
            }

            langHandle->getSyclQueue()->wait_and_throw();
            Tracer::func_end("BlasRoutines::scaleVector");   
            return 0; 
        }

        static int doMatMul(LangHandle *langHandle, int m, int k, int n, const float *d_A, const float *d_B, float *d_C) {
            Tracer::func_begin("BlasRoutines::doMatMul");
            
            float alpha = 1.0;
            float beta = 0.0;

            try {
                gemm(*(langHandle->getSyclQueue()), transpose::nontrans, transpose::nontrans, n, m, k, alpha, d_B, n, d_A,
                k, beta, d_C, n);
            }
            catch (sycl::exception const& e) {
                std::cout << "\t\tCaught synchronous SYCL exception during gemm:\n"
                    << e.what() << std::endl;
                    return 1;
            }
            catch (std::exception const& e) {
                std::cout << "\t\tCaught synchronous STL exception during gemm:\n"
                    << e.what() << std::endl;
                    return 2;
            }

            langHandle->getSyclQueue()->wait_and_throw();
            Tracer::func_end("BlasRoutines::doMatMul");     
            return 0; 
        }

        static int doMatMulTraA(LangHandle *langHandle, int m, int k, int n, const float *d_A, const float *d_B, float *d_C) {
            Tracer::func_begin("BlasRoutines::doMatMulTraA");
            float alpha = 1.0;
            float beta = 0.0;

            try {
                gemm(*(langHandle->getSyclQueue()), transpose::nontrans, transpose::trans, n, m, k, alpha, d_B, n, d_A,
                m, beta, d_C, n);
            }
            catch (sycl::exception const& e) {
                std::cout << "\t\tCaught synchronous SYCL exception during gemm:\n"
                    << e.what() << std::endl;
                    return 1;
            }
            catch (std::exception const& e) {
                std::cout << "\t\tCaught synchronous STL exception during gemm:\n"
                    << e.what() << std::endl;
                    return 2;
            }

            langHandle->getSyclQueue()->wait_and_throw();
            Tracer::func_end("BlasRoutines::doMatMulTraA");   
            return 0;   
        }

        static int doMatMulTraB(LangHandle *langHandle, int m, int k, int n, const float *d_A, const float *d_B, float *d_C) {
            Tracer::func_begin("BlasRoutines::doMatMulTranB");
            float alpha = 1.0;
            float beta = 0.0;

            try {
                gemm(*(langHandle->getSyclQueue()), transpose::trans, transpose::nontrans, n, m, k, alpha, d_B, k, d_A,
                k, beta, d_C, n);
            }
            catch (sycl::exception const& e) {
                std::cout << "\t\tCaught synchronous SYCL exception during gemm:\n"
                    << e.what() << std::endl;
                    return 1;
            }
            catch (std::exception const& e) {
                std::cout << "\t\tCaught synchronous STL exception during gemm:\n"
                    << e.what() << std::endl;
                    return 2;
            }

            langHandle->getSyclQueue()->wait_and_throw();
            Tracer::func_end("BlasRoutines::doMatMulTranB");     
            return 0; 
        }    
    #endif      
};

// class CublasRoutinesController {
//     public:
//         static void execute() {
//             Timer* timer = new Timer();

//             LangHandle *langHandle = new LangHandle(timer);

//             setupAndDoMatMul(langHandle);
//             setupAndDoMatMulTransA(langHandle);
//             setupAndDoMatMulTransB(langHandle);
//             //setupAndDoAxpy(langHandle);
//         }

//         static void setupAndDoMatMul(LangHandle *langHandle) {
//             int m = 3, k = 4, n = 5;

//             int aSize = m*k;
//             int bSize = k*n;
//             int cSize = m*n;

//             float *h_A = (float*)calloc(aSize, sizeof(float));  
//             float *h_B = (float*)calloc(bSize, sizeof(float));  
//             float *h_C = (float*)calloc(cSize, sizeof(float));  

//             float h_1A[12] = {1, 0, 4, 2,
//                               2, 3, 2, 1,
//                               2, 0, 1, 0};
//             float h_1B[20] = {3, 4, 0, 1, 2,
//                               3, 1, 0, 2, 1, 
//                               0, 3, 2, 4, 2, 
//                               2, 1, 3, 1, 2};

//             ImageProcessor::initImage(h_A, aSize);
//             ImageProcessor::initImage(h_B, bSize);

//             float *d_A, *d_B, *d_C;
//             assertDevApiInvar(cudaMalloc((void**)&(d_A), (aSize) * sizeof(float)));
//             assertDevApiInvar(cudaMalloc((void**)&(d_B), (bSize) * sizeof(float)));
//             assertDevApiInvar(cudaMalloc((void**)&(d_C), (cSize) * sizeof(float)));

//             assertDevApiInvar(cudaMemcpy(d_A, h_A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
//             assertDevApiInvar(cudaMemcpy(d_B, h_B, sizeof(float) * bSize, cudaMemcpyHostToDevice));

//             assertDevApiInvar(cudaMemcpy(d_A, &h_1A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
//             assertDevApiInvar(cudaMemcpy(d_B, &h_1B, sizeof(float) * bSize, cudaMemcpyHostToDevice));
//             assertDevApiInvar(// hipDeviceSynchronize());


//             assertBlasInvar(BlasRoutines::doMatMul(langHandle, m, k, n, d_A, d_B, d_C));

                      
//             assertDevApiInvar(cudaMemcpy(h_C, d_C, sizeof(float) * cSize, cudaMemcpyDeviceToHost));
//             assertDevApiInvar(// hipDeviceSynchronize());
//             for(int i=0; i<cSize; i++) {
//                 std::cout << h_C[i] << " " << std::endl; 
//             }
//             std::cout <<std::endl;
//         }



//         static void setupAndDoMatMulTransA(LangHandle *langHandle) {
//             int m = 3, k = 4, n = 5;

//             int aSize = m*k;
//             int bSize = k*n;
//             int cSize = m*n;

//             float *h_A = (float*)calloc(aSize, sizeof(float));  
//             float *h_B = (float*)calloc(bSize, sizeof(float));  
//             float *h_C = (float*)calloc(cSize, sizeof(float));  

//             // float h_1A[12] = {1, 0, 4, 2,
//             //                   2, 3, 2, 1,
//             //                   2, 0, 1, 0};
//             float h_1A[12] = {1, 2, 2,
//                               0, 3, 0,
//                               4, 2, 1,
//                               2, 1, 0};
                              
//             float h_1B[20] = {3, 4, 0, 1, 2,
//                               3, 1, 0, 2, 1, 
//                               0, 3, 2, 4, 2, 
//                               2, 1, 3, 1, 2};

//             ImageProcessor::initImage(h_A, aSize);
//             ImageProcessor::initImage(h_B, bSize);

//             float *d_A, *d_B, *d_C;
//             assertDevApiInvar(cudaMalloc((void**)&(d_A), (aSize) * sizeof(float)));
//             assertDevApiInvar(cudaMalloc((void**)&(d_B), (bSize) * sizeof(float)));
//             assertDevApiInvar(cudaMalloc((void**)&(d_C), (cSize) * sizeof(float)));

//             assertDevApiInvar(cudaMemcpy(d_A, h_A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
//             assertDevApiInvar(cudaMemcpy(d_B, h_B, sizeof(float) * bSize, cudaMemcpyHostToDevice));

//             assertDevApiInvar(cudaMemcpy(d_A, &h_1A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
//             assertDevApiInvar(cudaMemcpy(d_B, &h_1B, sizeof(float) * bSize, cudaMemcpyHostToDevice));
//             assertDevApiInvar(// hipDeviceSynchronize());



//             assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle, m, k, n, d_A, d_B, d_C));

                       

//             assertDevApiInvar(cudaMemcpy(h_C, d_C, sizeof(float) * cSize, cudaMemcpyDeviceToHost));
//             assertDevApiInvar(// hipDeviceSynchronize());
//             for(int i=0; i<cSize; i++) {
//                 std::cout << h_C[i] << " " << std::endl; 
//             }
//             std::cout <<std::endl;
//         }

//         static void setupAndDoMatMulTransB(LangHandle *langHandle) {
//             int m = 3, k = 4, n = 5;

//             int aSize = m*k;
//             int bSize = k*n;
//             int cSize = m*n;

//             float *h_A = (float*)calloc(aSize, sizeof(float));  
//             float *h_B = (float*)calloc(bSize, sizeof(float));  
//             float *h_C = (float*)calloc(cSize, sizeof(float));  

//             float h_1A[12] = {1, 0, 4, 2,
//                               2, 3, 2, 1,
//                               2, 0, 1, 0};
            
//             // float h_1B[20] = {3, 4, 0, 1, 2,
//             //                   3, 1, 0, 2, 1, 
//             //                   0, 3, 2, 4, 2, 
//             //                   2, 1, 3, 1, 2};
//             float h_1B[20] = {3, 3, 0, 2,
//                               4, 1, 3, 1,
//                               0, 0, 2, 3,
//                               1, 2, 4, 1,
//                               2, 1, 2, 2};                              

//             ImageProcessor::initImage(h_A, aSize);
//             ImageProcessor::initImage(h_B, bSize);

//             float *d_A, *d_B, *d_C;
//             assertDevApiInvar(cudaMalloc((void**)&(d_A), (aSize) * sizeof(float)));
//             assertDevApiInvar(cudaMalloc((void**)&(d_B), (bSize) * sizeof(float)));
//             assertDevApiInvar(cudaMalloc((void**)&(d_C), (cSize) * sizeof(float)));

//             assertDevApiInvar(cudaMemcpy(d_A, h_A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
//             assertDevApiInvar(cudaMemcpy(d_B, h_B, sizeof(float) * bSize, cudaMemcpyHostToDevice));

//             assertDevApiInvar(cudaMemcpy(d_A, &h_1A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
//             assertDevApiInvar(cudaMemcpy(d_B, &h_1B, sizeof(float) * bSize, cudaMemcpyHostToDevice));
//             assertDevApiInvar(// hipDeviceSynchronize());


//             assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle, m, k, n, d_A, d_B, d_C));


//             assertDevApiInvar(cudaMemcpy(h_C, d_C, sizeof(float) * cSize, cudaMemcpyDeviceToHost));
//             assertDevApiInvar(// hipDeviceSynchronize());
//             for(int i=0; i<cSize; i++) {
//                 std::cout << h_C[i] << " " << std::endl; 
//             }
//             std::cout <<std::endl;
//         }


//         static void setupAndDoAxpy(LangHandle *langHandle) {
//             int m = 2, n = 3;
//             int matSize = m*n;

//             float h_x[6] = {1, 0, 4,
//                             2, 3, 1};

//             float h_y[6] = {3, 1, 2,
//                             3, 5, 4,};                          

//             float *d_x, *d_y;
//             assertDevApiInvar(cudaMalloc((void**)&(d_x), (matSize) * sizeof(float)));
//             assertDevApiInvar(cudaMalloc((void**)&(d_y), (matSize) * sizeof(float)));
            
//             assertDevApiInvar(cudaMemcpy(d_x, &h_x, sizeof(float) * matSize, cudaMemcpyHostToDevice));
//             assertDevApiInvar(cudaMemcpy(d_y, &h_y, sizeof(float) * matSize, cudaMemcpyHostToDevice));
//             assertDevApiInvar(// hipDeviceSynchronize());

//             float scalingFactor = 1; 
//             assertBlasInvar(BlasRoutines::doAxpy(langHandle, matSize, &scalingFactor, d_x, d_y));

//             assertDevApiInvar(cudaMemcpy(&h_y, d_y, sizeof(float) * matSize, cudaMemcpyDeviceToHost));
//             assertDevApiInvar(// hipDeviceSynchronize());
//             for(int i=0; i<matSize; i++) {
//                 std::cout << h_y[i] << " " << std::endl; 
//             }
//             std::cout <<std::endl;
//         }
// };

#endif