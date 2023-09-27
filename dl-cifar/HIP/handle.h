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

#ifndef DL_CIFAR_HANDLE_H_
#define DL_CIFAR_HANDLE_H_

#include <iostream>
#include <string>
#include <miopen/miopen.h>
#include <rocblas.h>
#include "timing.h"

#include "error_handling.h"

using namespace dl_cifar::common;

enum MemcpyType {
    H2D,
    D2H,
    D2D
};

class LangHandle {
    private:
        Timer* timer_;
        miopenHandle_t *miopenHandle_;
        rocblas_handle *rocblasHandle_;

    public:
        LangHandle(Timer* timer): timer_(timer) {
            assertDevApiInvar(hipSetDevice(0));

#ifdef DEVICE_TIMER    
        Time start = get_time_now();
#endif                
            miopenHandle_ = new miopenHandle_t();
            assertDlApiInvar(miopenCreate(miopenHandle_));
#ifdef DEVICE_TIMER       
        timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_CUDNN_HANDLE");
#endif

#ifdef DEVICE_TIMER    
        Time start = get_time_now();
#endif  
            rocblasHandle_ = new rocblas_handle();
            rocblas_create_handle(rocblasHandle_); 
#ifdef DEVICE_TIMER       
        timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_CUBLAS_HANDLE");
#endif            
        }

        miopenHandle_t* getMiopenHandle() {
            return miopenHandle_;
        }

        rocblas_handle* getRocblasHandle() {
            return rocblasHandle_;
        }

        float* allocDevMem(size_t size) {
            float *d_ptr;
            assertDevApiInvar(hipMalloc((void**)&(d_ptr), size));
            return d_ptr;
        }

        void freeDevPtr(float* devPtr) {
            assertDevApiInvar(hipFree(devPtr));    
        }

        void memCpy(float* devPtr, const float* hostPtr, size_t size, bool needToSynchronize, MemcpyType memcpyType) {
            if(memcpyType == H2D) {
                memCpyH2D(devPtr, hostPtr, size, needToSynchronize);
            } else if(memcpyType == D2H) {
                memCpyD2H(devPtr, hostPtr, size, needToSynchronize);
            } else if(memcpyType == D2D) {
                memCpyD2D(devPtr, hostPtr, size, needToSynchronize);
            } else {
	        throw std::runtime_error("Unknown or unsupported MemcpyType");
	    }

        }

        void memCpyH2D(float* devPtr, const float* hostPtr, size_t size, bool needToSynchronize) {
            assertDevApiInvar(hipMemcpy(devPtr, hostPtr, size, hipMemcpyHostToDevice));
            if(needToSynchronize) {
                assertDevApiInvar(hipDeviceSynchronize());
            }
        }

        void memCpyD2D(float* devPtr1, const float* devPtr2, size_t size, bool needToSynchronize) {
            assertDevApiInvar(hipMemcpy(devPtr1, devPtr2, size, hipMemcpyDeviceToDevice));
            if(needToSynchronize) {
                assertDevApiInvar(hipDeviceSynchronize());
            }
        }

        void memCpyD2H(float* hostPtr, const float* devPtr, size_t size, bool needToSynchronize) {
            assertDevApiInvar(hipMemcpy(hostPtr, devPtr, size, hipMemcpyDeviceToHost));
            if(needToSynchronize) {
                assertDevApiInvar(hipDeviceSynchronize());
            }
        }

        void synchronize() {
            assertDevApiInvar(hipDeviceSynchronize());
        }

};


#endif

