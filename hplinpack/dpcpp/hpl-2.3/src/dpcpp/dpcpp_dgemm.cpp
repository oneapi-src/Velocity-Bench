  /* 
 * -- High Performance Computing Linpack Benchmark (HPL)                
 *    Modifications Copyright (C) 2023 Intel Corporation​
 *                                                                      
 * -- Copyright notice and Licensing terms:                             
 *                                                                      
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:                                                             
 *                                                                      
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.        
 *                                                                      
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. 
 *                                                                      
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:                 
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.             
 *                                                                      
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.                                                          
 *                                                                      
 * -- Disclaimer:                                                       
 *                                                                      
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 * ---------------------------------------------------------------------
 * 
 * SPDX-License-Identifier: BSD-4-Clause
 */ 

/* 
 * -- High Performance Computing Linpack Benchmark (HPL)                
 *    HPL - 2.3 - December 2, 2018                          
 *    Antoine P. Petitet                                                
 *    University of Tennessee, Knoxville                                
 *    Innovative Computing Laboratory                                 
 *    (C) Copyright 2000-2008 All Rights Reserved                       
 *                                                                      
 * -- Copyright notice and Licensing terms:                             
 *                                                                      
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:                                                             
 *                                                                      
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.        
 *                                                                      
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. 
 *                                                                      
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:                 
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.             
 *                                                                      
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.                                                          
 *                                                                      
 * -- Disclaimer:                                                       
 *                                                                      
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 * ---------------------------------------------------------------------
 */ 



#define NUMBER_OF_STREAMS 4
#define CHUNK_SIZE 512
#define NN 64
#define NM 128
//#define DPCPP_DEBUG
//#define DEVICE_DEBUG
//#define MPI

#ifdef MPI
#include <mpi.h>
#endif

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <dlfcn.h>
#include <ctype.h>
#include <math.h>
#include <array>

#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>

#include <iostream>
#include <chrono> 
#include <sycl/sycl.hpp>
#include "oneapi/mkl/blas.hpp"
#include "dpcpp_dgemm.h"


#ifdef USE_CUBLAS
#include <sycl/backend/cuda.hpp>
#include <cublas_v2.h>
#include <cuda.h>
#include "mkl.h"
#include <cuda_runtime.h>
#elif USE_HIPBLAS
#include "hipblas.h"
#else
#include "oneapi/mkl/blas.hpp"
#endif

extern "C" {
    void dpcpp_dgemm 
        (   const int ORDER,
            const int TRANSA,   const int TRANSB,
            const int M,        const int N,        const int K,       
            const double ALPHA, const double *A,    const int LDA,
            const double *B,    const int LDB,      const double BETA,    
            double *C,          const int LDC);

    void dpcpp_dtrsm(
       int HPL_ORDER,
       int HPL_SIDE,
       int HPL_UPLO,
       int HPL_TRANS,
       int HPL_DIAG,
       const int,
       const int,
       const double,
       const double *,
       const int,
       double *,
       const int);
}

void DeviceManager::display_device_properties(sycl::device const &dev)
{
    std::cout << "\tSYCL device              : " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "\tDriver version           : " << dev.get_info<sycl::info::device::driver_version>() << std::endl;
    std::cout << "\tPlatform                 : " << dev.get_platform().get_info<sycl::info::platform::name>()<< std::endl;
    std::cout << "\tVendor                   : " << dev.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "\tMax compute units        : " << dev.get_info<sycl::info::device::max_compute_units>() << std::endl;
}

#ifdef USE_CUBLAS
#define CHECK_ERROR(FUNC) checkCudaErrorMsg(FUNC, " " #FUNC)

void inline checkCudaErrorMsg(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "ERROR CUBLAS:" << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

void inline checkCudaErrorMsg(cudaError status, const char *msg) {
  if (status != cudaSuccess) {
    std::cout << "ERROR CUDA: " << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

void inline checkCudaErrorMsg(CUresult status, const char *msg) {
  if (status != CUDA_SUCCESS) {
    std::cout << "ERROR CUDA: " << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

#endif

// helper functions to determine buffer dimension
template <typename T> constexpr T inner_dimension(oneapi::mkl::transpose trans, T m, T n)
    { return (trans == oneapi::mkl::transpose::nontrans) ? m : n; }
template <typename T> constexpr T outer_dimension(oneapi::mkl::transpose trans, T m, T n)
    { return (trans == oneapi::mkl::transpose::nontrans) ? n : m; }
template <typename T> constexpr T matrix_size(oneapi::mkl::transpose trans, T m, T n, T ldm)
    {   return outer_dimension(trans, m, n) * ldm; 
       //return outer_dimension(trans, m, n);
    }

// TODO: hardcoded values for enums, 
inline oneapi::mkl::transpose to_mkl_trans(int hpltrans){
    if(hpltrans==111) return oneapi::mkl::transpose::nontrans;
    if(hpltrans==112) return oneapi::mkl::transpose::trans;
    if(hpltrans==113) return oneapi::mkl::transpose::conjtrans;
    return oneapi::mkl::transpose::trans;
}
    
inline oneapi::mkl::uplo   to_mkl_uplo(int hpluplo){
    if(hpluplo==121) return oneapi::mkl::uplo::upper;
    if(hpluplo==122) return oneapi::mkl::uplo::lower;
    return oneapi::mkl::uplo::upper;
}
    
inline oneapi::mkl::diag to_mkl_diag(int hpldiag){
    if(hpldiag==131) return oneapi::mkl::diag::nonunit;
    if(hpldiag==132) return oneapi::mkl::diag::unit;
    return oneapi::mkl::diag::nonunit;
}

inline oneapi::mkl::side to_mkl_side(int hplside){
    if(hplside==141) return oneapi::mkl::side::left;
    if(hplside==142) return oneapi::mkl::side::right;
    return oneapi::mkl::side::left;
}
void dpcpp_dgemm 
(   const int ORDER,   const int TRANSA,    const int TRANSB,       
    const int M,       const int N,         const int K,       
    const double ALPHA,const double *A,     const int LDA,
    const double *B,   const int LDB,       
    const double BETA, double *C,         const int LDC)
{



if ((M==0)||(K==0)||(N==0))
        return;



#ifdef DPCPP_DEBUG    
    using namespace std;
    cout <<"Calling DPC++ dgemm ========="<<endl;
    cout << "order  "<< ORDER << endl;
    cout << "M      "<< M << endl;
    cout << "N      "<< N << endl;
    cout << "K      "<< K << endl;
    cout << "A      "<< A << endl;
    cout << "B      "<< B << endl;
    cout << "C      "<< C << endl;
    cout << "ALPHA  "<< ALPHA << endl;
    cout << "BETA   "<< BETA << endl;
    cout << "LDA    "<< LDA << endl;
    cout << "LDB    "<< LDB << endl;
    cout << "LDC    "<< LDC << endl;
    cout <<"=============================="<<endl;
#endif    
    oneapi::mkl::transpose transA = to_mkl_trans(TRANSA);
    oneapi::mkl::transpose transB = to_mkl_trans(TRANSB); 

  
    if ( (N) < NN || (M) < NM || (K) < 128){  
      
      #ifdef DEVICE_DEBUG
   	std::cout << "gemm-CPU\n";
      #endif
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
              
      return;
    }      

    int id = 0;
    #ifdef MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    #endif

    #ifdef DEVICE_DEBUG
        std::cout << "gemm-GPU\n";
    #endif

    
    sycl::queue mQueue{};
    auto ctxt = mQueue.get_context();
    auto mdevice = mQueue.get_device(); 

   
   
    double *A_buffer = (double *)malloc_device(matrix_size(transA, M, K, LDA) * sizeof(double), mdevice, ctxt);
    mQueue.memcpy(A_buffer, A,  matrix_size(transA, M, K, LDA) * sizeof(double));
   
    int b_size_total = matrix_size(transB, K, N, LDB);
    int c_size_total = matrix_size(oneapi::mkl::transpose::nontrans, M, N, LDC);	

    double *B_buffer = (double *)malloc_device(b_size_total * sizeof(double), mdevice, ctxt);
    mQueue.memcpy(B_buffer, B,  b_size_total * sizeof(double));
    
   
    double *C_buffer = (double *)malloc_device(c_size_total * sizeof(double), mdevice, ctxt);
    mQueue.memcpy(C_buffer, C,  c_size_total * sizeof(double));

    mQueue.wait();	    
    
    #ifdef USE_CUBLAS
    cublasHandle_t handle;
    CHECK_ERROR(cublasCreate(&handle)); 

    mQueue.submit([&](sycl::handler &h){

                h.host_task([=](sycl::interop_handle ih) {
                      cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                      cublasSetStream(handle, ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());
		      
		      CHECK_ERROR(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &ALPHA, A_buffer, LDA, B_buffer, LDB, &BETA, C_buffer, LDC));
              cudaDeviceSynchronize ();	
		});
	}).wait_and_throw();
    #elif USE_HIPBLAS
       hipblasHandle_t handle;
        hipblasCreate(&handle);


	mQueue.submit([&](sycl::handler &h){

                h.host_task([=](sycl::interop_handle ih) {
                      hipCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_hip>());
                      hipblasSetStream(handle, ih.get_native_queue<sycl::backend::ext_oneapi_hip>());
		      
		       hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K, &ALPHA, A_buffer, LDA, B_buffer, LDB, &BETA, C_buffer, LDC);	
		});
	}).wait_and_throw();
    #else
    oneapi::mkl::blas::gemm(mQueue, transA, transB, M, N, K, ALPHA, A_buffer, LDA, B_buffer, LDB, BETA, C_buffer, LDC);
    mQueue.wait();
    #endif
    mQueue.memcpy(C, C_buffer, c_size_total * sizeof(double)).wait(); 
    free(A_buffer, mQueue);
    free(B_buffer, mQueue);	
    free(C_buffer, mQueue);
}
  
void dpcpp_dtrsm

(  const int ORDER,           const int SIDE,
   const int UPLO,            const int TRANS,
   const int DIAG,            const int M,       const int N,
   const double ALPHA,    const double* A,  const int LDA,       double* B,
   const int LDB)
{

  if ((M==0)||(N==0)){
        return;
  }



#ifdef DPCPP_DEBUG    
    using namespace std;
    cout <<"Calling DPC++ dtrsm ========="<<endl;
    cout << "ORDER      "<< ORDER << endl;
    cout << "SIDE       "<< SIDE << endl;
    cout << "UPLO       "<< UPLO << endl;
    cout << "TRANS      "<< TRANS << endl;
    cout << "DIAG       "<< DIAG << endl;
    cout << "M          "<< M << endl;
    cout << "N          "<< N << endl;
    cout << "A          "<< A << endl;
    cout << "LDA        "<< LDA << endl;
    cout << "B          "<< B << endl;
    cout << "LDB        "<< LDB << endl;
    cout <<"============================="<<endl;
#endif    
    
    oneapi::mkl::side      side     = to_mkl_side(SIDE);
    oneapi::mkl::uplo      uplo     = to_mkl_uplo(UPLO);
    oneapi::mkl::transpose trans    = to_mkl_trans(TRANS);
    oneapi::mkl::diag      diag     = to_mkl_diag(DIAG);
    
   
    if ( (M) < 512 || (N) < 2*(M)){

        #ifdef DEVICE_DEBUG
         std::cout << "dtrsm-CPU\n";
        #endif 	 

        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, M, N, ALPHA, A, LDA, B, LDB);
     	return;
    }
   
     
    #ifdef DEVICE_DEBUG
        std::cout << "dtrsm-GPU\n";
    #endif

    int id = 0;
    #ifdef MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &id); 
    #endif	       

    sycl::queue mQueue{};
    auto ctxt = mQueue.get_context();
    auto mdevice = mQueue.get_device(); 

    
    double *A_buffer = (double *)malloc_device(M * LDA  * sizeof(double), mdevice, ctxt);
    mQueue.memcpy(A_buffer, A,  M * LDA * sizeof(double));

    double *B_buffer = (double *)malloc_device(N * LDB  * sizeof(double), mdevice, ctxt);
    mQueue.memcpy(B_buffer, B,  N * LDB * sizeof(double));

    mQueue.wait(); 
    #ifdef USE_CUBLAS 
    cublasHandle_t handle;
    CHECK_ERROR(cublasCreate(&handle)); 
    //constexpr double CU_ALPHA = ALPHA;

	mQueue.submit([&](sycl::handler &h){

                h.host_task([=](sycl::interop_handle ih) {
                      cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
                      cublasSetStream(handle, ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());
		      
		      CHECK_ERROR(cublasDtrsm(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_UNIT,M,N,&ALPHA,A_buffer,LDA,B_buffer,LDB));
              cudaDeviceSynchronize();	
		});
	}).wait_and_throw();		
    #elif USE_HIPBLAS
    hipblasHandle_t handle;
        hipblasCreate(&handle); 

        
	mQueue.submit([&](sycl::handler &h){
                h.host_task([=](sycl::interop_handle ih) {
		      hipCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_hip>());
                      hipblasSetStream(handle, ih.get_native_queue<sycl::backend::ext_oneapi_hip>());                      

		      hipblasDtrsm(handle,HIPBLAS_SIDE_LEFT,HIPBLAS_FILL_MODE_LOWER,HIPBLAS_OP_N,HIPBLAS_DIAG_UNIT,M,N,&ALPHA,A_buffer,LDA,B_buffer,LDB);	
		});
	}).wait_and_throw(); 
     
    #else

    oneapi::mkl::blas::trsm(mQueue, side, uplo, trans, diag, M, N, ALPHA, A_buffer, LDA, B_buffer, LDB);
    mQueue.wait();


    #endif
    
    mQueue.memcpy(B, B_buffer, N * LDB * sizeof(double)).wait();
         
    free(A_buffer, mQueue);
    free(B_buffer, mQueue);
}
