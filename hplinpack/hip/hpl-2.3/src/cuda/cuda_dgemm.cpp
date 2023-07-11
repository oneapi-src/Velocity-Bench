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
#define ERRCODE(e) (-(__LINE__ * 1000 + (e)))
//#define DEVICE_DEBUG
//#ifdef MPI
//#include <mpi.h>
//#endif


#define _GNU_SOURCE

//#define CUDA_ERROR_CHECK
//#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
//#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <dlfcn.h>
#include <ctype.h>
#include <math.h>
#include <array>

#include <time.h>

#include <iostream>
#include <chrono> 

/*
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
*/
#include "hip/hip_runtime.h" 

#include "mkl.h"
#include "hipblas.h"

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


void dpcpp_dgemm 
(   const int ORDER,   const int TRANSA,    const int TRANSB,       
    const int M,       const int N,         const int K,       
    const double ALPHA,const double *A,     const int LDA,
    const double *B,   const int LDB,       
    const double BETA, double *C,         const int LDC)
{

    if ((M==0)||(K==0)||(N==0)){
	    return;
    }

    
    if ( (N) < NN || (M) < NM || (K) < 128){ 
         
         #ifdef DEVICE_DEBUG
            std::cout << "dgemm-Running on CPU" << std::endl; 
         #endif
          
         cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
          return;
    }    

    
    #ifdef DEVICE_DEBUG
            std::cout << "dgemm-Running on GPU" << std::endl; 
    #endif

    double *devPtrA, *devPtrB, *devPtrC;
    //int status;
    
    hipblasHandle_t handle = NULL;
    
     
    hipblasStatus_t status;
    status = hipblasCreate(&handle);
    

    hipMalloc((void **)&devPtrA, K * LDA *sizeof(double));
    hipMemcpy(devPtrA, &A[0], K * LDA *sizeof(double), hipMemcpyHostToDevice);
    
    hipMalloc((void **)&devPtrB, N *  LDB *sizeof(double));
    hipMemcpy(devPtrB, &B[0], N *  LDB *sizeof(double), hipMemcpyHostToDevice);
    
    hipMalloc((void **)&devPtrC, N * LDC *sizeof(double));
    hipMemcpy(devPtrC, &C[0], N * LDC *sizeof(double), hipMemcpyHostToDevice);
    

    hipDeviceSynchronize();
    hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K, &ALPHA, devPtrA, LDA, devPtrB, LDB, &BETA, devPtrC, LDC);
    hipDeviceSynchronize();
    hipMemcpy(&C[0], devPtrC, N * LDC *sizeof(double), hipMemcpyDeviceToHost);
    hipDeviceSynchronize(); 
    hipFree(devPtrA);
    hipFree(devPtrB);
    hipFree(devPtrC);
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

    double *devPtrA, *devPtrB;	
    //int status;	

    
    if ( (M) < 512 || (N) < 2*(M)){
        #ifdef DEVICE_DEBUG
            std::cout << "dtrsm-Running on CPU" << std::endl; 
        #endif
 	    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, M, N, ALPHA, A, LDA, B, LDB);
    
    
    	return;
    } 
       
    #ifdef DEVICE_DEBUG
            std::cout << "dtrsm-Running on GPU" << std::endl; 
    #endif

    hipblasHandle_t handle = NULL;
    
    
    hipblasStatus_t status;
    status = hipblasCreate(&handle);
    

    hipMalloc((void **)&devPtrA, M * LDA * sizeof(double));
    hipMemcpy(devPtrA, A, M * LDA * sizeof(double), hipMemcpyHostToDevice);   	
    
        
    hipMalloc((void **)&devPtrB, N * LDB * sizeof(double));
    hipMemcpy(devPtrB, B, N * LDB * sizeof(double), hipMemcpyHostToDevice);
    hipDeviceSynchronize(); 
   
    hipblasDtrsm(handle,HIPBLAS_SIDE_LEFT,HIPBLAS_FILL_MODE_LOWER,HIPBLAS_OP_N,HIPBLAS_DIAG_UNIT,M,N,&ALPHA,devPtrA,LDA,devPtrB,LDB);
    
    hipDeviceSynchronize();     
    hipMemcpy(B, devPtrB, N * LDB * sizeof(double), hipMemcpyDeviceToHost);
    
    hipDeviceSynchronize();
    hipFree(devPtrA);
    hipFree(devPtrB);
  
        
}
