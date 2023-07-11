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

#include <cublas_v2.h>
#include <iostream>
#include "error_handling.h"
#include "tracing.h"
#include "handle.h"
#include "image_processing.h"

using namespace dl_cifar::common;

class BlasRoutines {
    public:

        static cublasStatus_t doAxpy(LangHandle *langHandle, int noOfElems, float *scalingFactor, float *d_x, float *d_y) {
            Tracer::func_begin("MklRoutines::doAxpy");    

            cublasStatus_t status = cublasSaxpy(*(langHandle->getCublasHandle()), noOfElems, scalingFactor, d_x, 1, d_y, 1);
            langHandle->synchronize();

            Tracer::func_end("MklRoutines::doAxpy");  
            return status;
        }


        static cublasStatus_t scaleVector(LangHandle *langHandle, int noOfElements, float *scalingFactor, float* vector, int inc) {
            Tracer::func_begin("MklRoutines::scaleVector");    

            cublasStatus_t status = cublasSscal(*(langHandle->getCublasHandle()), noOfElements, scalingFactor, vector, inc);
            langHandle->synchronize();
            Tracer::func_end("MklRoutines::scaleVector");  
            return status;
        }
        
        static cublasStatus_t doMatMul(LangHandle *langHandle, int m, int k, int n, const float *A, const float *B, float *C) {
            Tracer::func_begin("MklRoutines::doMatMul");    

            float alpha_ = 1.0;
            float beta_ = 0.0;
            cublasStatus_t status =  cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_N,
                                                    n, m, k, &alpha_, B, n, A, k, &beta_, C, n);
            langHandle->synchronize();                                        
            Tracer::func_end("MklRoutines::doMatMul");  
            return status;                                        
        }

    

        static cublasStatus_t doMatMulTraA(LangHandle *langHandle, int m, int k, int n, float *A, const float *B, float *C) {
            Tracer::func_begin("MklRoutines::doMatMulTraA");    

            float alpha_ = 1.0;
            float beta_ = 0.0;
            cublasStatus_t status =  cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_T,
                                                    n, m, k, &alpha_, B, n, A, m, &beta_, C, n);
            langHandle->synchronize();                                        
            Tracer::func_end("MklRoutines::doMatMulTraA");  
            return status;                                        
        }


        static cublasStatus_t doMatMulTraB(LangHandle *langHandle, int m, int k, int n, float *A, const float *B, float *C) {
            Tracer::func_begin("MklRoutines::doMatMulTraB");    

            float alpha_ = 1.0;
            float beta_ = 0.0;
            cublasStatus_t status =  cublasSgemm(*(langHandle->getCublasHandle()), CUBLAS_OP_T, CUBLAS_OP_N,
                                                    n, m, k, &alpha_, B, k, A, k, &beta_, C, n);
            langHandle->synchronize();                                                    
            Tracer::func_end("MklRoutines::doMatMulTraB");  
            return status;                                        
        }

    
        static cublasStatus_t doBatchedMatMul(LangHandle *langHandle, int m, int k, int n, 
                                                const float *A[], const float *B[], float *C[], int batchCount) {
            Tracer::func_begin("MklRoutines::doBatchedMatMul");    

            float alpha_ = 1.0;
            float beta_ = 0.0;

            cublasStatus_t status =  cublasSgemmBatched(*(langHandle->getCublasHandle()), CUBLAS_OP_N, CUBLAS_OP_N,
                                                    n, m, k, &alpha_, B, n, A, k, &beta_, C, n, batchCount);
            langHandle->synchronize();                                         
            Tracer::func_end("MklRoutines::doBatchedMatMul");  
            return status;                                        
        }              
};

class CublasRoutinesController {
    public:
        static void execute() {
            Timer* timer = new Timer();

            LangHandle *langHandle = new LangHandle(timer);

            setupAndDoMatMul(langHandle);
            setupAndDoMatMulTransA(langHandle);
            setupAndDoMatMulTransB(langHandle);
            //setupAndDoAxpy(langHandle);
        }

        static void setupAndDoMatMul(LangHandle *langHandle) {
            int m = 3, k = 4, n = 5;

            int aSize = m*k;
            int bSize = k*n;
            int cSize = m*n;

            float *h_A = (float*)calloc(aSize, sizeof(float));  
            float *h_B = (float*)calloc(bSize, sizeof(float));  
            float *h_C = (float*)calloc(cSize, sizeof(float));  

            float h_1A[12] = {1, 0, 4, 2,
                              2, 3, 2, 1,
                              2, 0, 1, 0};
            float h_1B[20] = {3, 4, 0, 1, 2,
                              3, 1, 0, 2, 1, 
                              0, 3, 2, 4, 2, 
                              2, 1, 3, 1, 2};

            ImageProcessor::initImage(h_A, aSize);
            ImageProcessor::initImage(h_B, bSize);

            float *d_A, *d_B, *d_C;
            assertDevApiInvar(cudaMalloc((void**)&(d_A), (aSize) * sizeof(float)));
            assertDevApiInvar(cudaMalloc((void**)&(d_B), (bSize) * sizeof(float)));
            assertDevApiInvar(cudaMalloc((void**)&(d_C), (cSize) * sizeof(float)));

            assertDevApiInvar(cudaMemcpy(d_A, h_A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
            assertDevApiInvar(cudaMemcpy(d_B, h_B, sizeof(float) * bSize, cudaMemcpyHostToDevice));

            assertDevApiInvar(cudaMemcpy(d_A, &h_1A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
            assertDevApiInvar(cudaMemcpy(d_B, &h_1B, sizeof(float) * bSize, cudaMemcpyHostToDevice));
            assertDevApiInvar(cudaDeviceSynchronize());


            assertBlasInvar(BlasRoutines::doMatMul(langHandle, m, k, n, d_A, d_B, d_C));

                      
            assertDevApiInvar(cudaMemcpy(h_C, d_C, sizeof(float) * cSize, cudaMemcpyDeviceToHost));
            assertDevApiInvar(cudaDeviceSynchronize());
            for(int i=0; i<cSize; i++) {
                std::cout << h_C[i] << " " << std::endl; 
            }
            std::cout <<std::endl;
        }



        static void setupAndDoMatMulTransA(LangHandle *langHandle) {
            int m = 3, k = 4, n = 5;

            int aSize = m*k;
            int bSize = k*n;
            int cSize = m*n;

            float *h_A = (float*)calloc(aSize, sizeof(float));  
            float *h_B = (float*)calloc(bSize, sizeof(float));  
            float *h_C = (float*)calloc(cSize, sizeof(float));  

            // float h_1A[12] = {1, 0, 4, 2,
            //                   2, 3, 2, 1,
            //                   2, 0, 1, 0};
            float h_1A[12] = {1, 2, 2,
                              0, 3, 0,
                              4, 2, 1,
                              2, 1, 0};
                              
            float h_1B[20] = {3, 4, 0, 1, 2,
                              3, 1, 0, 2, 1, 
                              0, 3, 2, 4, 2, 
                              2, 1, 3, 1, 2};

            ImageProcessor::initImage(h_A, aSize);
            ImageProcessor::initImage(h_B, bSize);

            float *d_A, *d_B, *d_C;
            assertDevApiInvar(cudaMalloc((void**)&(d_A), (aSize) * sizeof(float)));
            assertDevApiInvar(cudaMalloc((void**)&(d_B), (bSize) * sizeof(float)));
            assertDevApiInvar(cudaMalloc((void**)&(d_C), (cSize) * sizeof(float)));

            assertDevApiInvar(cudaMemcpy(d_A, h_A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
            assertDevApiInvar(cudaMemcpy(d_B, h_B, sizeof(float) * bSize, cudaMemcpyHostToDevice));

            assertDevApiInvar(cudaMemcpy(d_A, &h_1A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
            assertDevApiInvar(cudaMemcpy(d_B, &h_1B, sizeof(float) * bSize, cudaMemcpyHostToDevice));
            assertDevApiInvar(cudaDeviceSynchronize());



            assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle, m, k, n, d_A, d_B, d_C));

                       

            assertDevApiInvar(cudaMemcpy(h_C, d_C, sizeof(float) * cSize, cudaMemcpyDeviceToHost));
            assertDevApiInvar(cudaDeviceSynchronize());
            for(int i=0; i<cSize; i++) {
                std::cout << h_C[i] << " " << std::endl; 
            }
            std::cout <<std::endl;
        }

        static void setupAndDoMatMulTransB(LangHandle *langHandle) {
            int m = 3, k = 4, n = 5;

            int aSize = m*k;
            int bSize = k*n;
            int cSize = m*n;

            float *h_A = (float*)calloc(aSize, sizeof(float));  
            float *h_B = (float*)calloc(bSize, sizeof(float));  
            float *h_C = (float*)calloc(cSize, sizeof(float));  

            float h_1A[12] = {1, 0, 4, 2,
                              2, 3, 2, 1,
                              2, 0, 1, 0};
            
            // float h_1B[20] = {3, 4, 0, 1, 2,
            //                   3, 1, 0, 2, 1, 
            //                   0, 3, 2, 4, 2, 
            //                   2, 1, 3, 1, 2};
            float h_1B[20] = {3, 3, 0, 2,
                              4, 1, 3, 1,
                              0, 0, 2, 3,
                              1, 2, 4, 1,
                              2, 1, 2, 2};                              

            ImageProcessor::initImage(h_A, aSize);
            ImageProcessor::initImage(h_B, bSize);

            float *d_A, *d_B, *d_C;
            assertDevApiInvar(cudaMalloc((void**)&(d_A), (aSize) * sizeof(float)));
            assertDevApiInvar(cudaMalloc((void**)&(d_B), (bSize) * sizeof(float)));
            assertDevApiInvar(cudaMalloc((void**)&(d_C), (cSize) * sizeof(float)));

            assertDevApiInvar(cudaMemcpy(d_A, h_A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
            assertDevApiInvar(cudaMemcpy(d_B, h_B, sizeof(float) * bSize, cudaMemcpyHostToDevice));

            assertDevApiInvar(cudaMemcpy(d_A, &h_1A, sizeof(float) * aSize, cudaMemcpyHostToDevice));
            assertDevApiInvar(cudaMemcpy(d_B, &h_1B, sizeof(float) * bSize, cudaMemcpyHostToDevice));
            assertDevApiInvar(cudaDeviceSynchronize());


            assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle, m, k, n, d_A, d_B, d_C));


            assertDevApiInvar(cudaMemcpy(h_C, d_C, sizeof(float) * cSize, cudaMemcpyDeviceToHost));
            assertDevApiInvar(cudaDeviceSynchronize());
            for(int i=0; i<cSize; i++) {
                std::cout << h_C[i] << " " << std::endl; 
            }
            std::cout <<std::endl;
        }


        static void setupAndDoAxpy(LangHandle *langHandle) {
            int m = 2, n = 3;
            int matSize = m*n;

            float h_x[6] = {1, 0, 4,
                            2, 3, 1};

            float h_y[6] = {3, 1, 2,
                            3, 5, 4,};                          

            float *d_x, *d_y;
            assertDevApiInvar(cudaMalloc((void**)&(d_x), (matSize) * sizeof(float)));
            assertDevApiInvar(cudaMalloc((void**)&(d_y), (matSize) * sizeof(float)));
            
            assertDevApiInvar(cudaMemcpy(d_x, &h_x, sizeof(float) * matSize, cudaMemcpyHostToDevice));
            assertDevApiInvar(cudaMemcpy(d_y, &h_y, sizeof(float) * matSize, cudaMemcpyHostToDevice));
            assertDevApiInvar(cudaDeviceSynchronize());

            float scalingFactor = 1; 
            assertBlasInvar(BlasRoutines::doAxpy(langHandle, matSize, &scalingFactor, d_x, d_y));

            assertDevApiInvar(cudaMemcpy(&h_y, d_y, sizeof(float) * matSize, cudaMemcpyDeviceToHost));
            assertDevApiInvar(cudaDeviceSynchronize());
            for(int i=0; i<matSize; i++) {
                std::cout << h_y[i] << " " << std::endl; 
            }
            std::cout <<std::endl;
        }
};

#endif