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
#include "timing.h"
#include <sycl/sycl.hpp>

#if defined(USE_CUBLAS)
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "error_handling.h"
#elif defined(USE_ROCBLAS) 
#include <miopen/miopen.h>
#include "hip/hip_runtime_api.h"
#include <rocblas.h>
#include "error_handling.h"
#else
#include "oneapi/dnnl/dnnl.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "dnnl_sycl.hpp"
#include <oneapi/mkl.hpp>
using namespace dnnl;
#endif
#endif
using namespace dl_cifar::common;


enum MemcpyType {
    H2D,
    D2H,
    D2D
};

#if defined(USE_CUBLAS)
class LangHandle {
    private:
        Timer* timer_;
        cudnnHandle_t  *cudnnHandle_;
        cublasHandle_t *cublasHandle_;

        sycl::device *dht_;
        sycl::context *context_;
        sycl::queue *sycl_queue_;  
        //engine eng_;   
        //stream s_;

    public:
        LangHandle(Timer* timer): timer_(timer) {
            assertDevApiInvar(cudaSetDevice(0));

#ifdef DEVICE_TIMER    
        Time start = get_time_now();
#endif                
            cudnnHandle_ = new cudnnHandle_t();
            assertDlApiInvar(cudnnCreate(cudnnHandle_));
#ifdef DEVICE_TIMER       
        timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_CUDNN_HANDLE");
#endif

#ifdef DEVICE_TIMER    
        Time start = get_time_now();
#endif  
            cublasHandle_ = new cublasHandle_t();
            cublasCreate(cublasHandle_); 
#ifdef DEVICE_TIMER       
        timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_CUBLAS_HANDLE");
#endif 




            //dht_ = new sycl::device(sycl::gpu_selector_v); 
            dht_ = new sycl::device(sycl::gpu_selector_v); 
#ifdef DEVICE_TIMER  
            start = get_time_now();
#endif    
            context_ = new sycl::context(*dht_);
#ifdef DEVICE_TIMER  
            timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_SYCL_CONTEXT");
#endif    
            auto propList = sycl::property_list{
                #ifdef IN_ORDER_QUEUE
                sycl::property::queue::in_order{},
                #ifdef SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
                sycl::ext::oneapi::property::queue::discard_events{}
                #endif
                #endif
            };
#ifdef DEVICE_TIMER  
            start = get_time_now();  
#endif     
            sycl_queue_ = new sycl::queue(*context_, *dht_, propList);
#ifdef DEVICE_TIMER  
            timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_SYCL_QUEUE");
#endif    
// #ifdef DEVICE_TIMER  
//             start = get_time_now();
// #endif    
//             //engine eng(engine::kind::gpu, 0);
//             eng_ = dnnl::sycl_interop::make_engine(*dht_, *context_);
// #ifdef DEVICE_TIMER  
//             timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_ONEDNN_ENGINE");
// #endif    
// #ifdef DEVICE_TIMER      
//             start = get_time_now();
// #endif    
//             //stream s(eng);
//             s_ = dnnl::sycl_interop::make_stream(eng_, *sycl_queue_);
// #ifdef DEVICE_TIMER      
//             imer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_ONEDNN STREAM");
// #endif    

        }

        sycl::device* getDevice() {
            return dht_;
        }

        sycl::context* getContext() {
            return context_;
        }

        sycl::queue* getSyclQueue() {
            return sycl_queue_;
        }

        // engine* getEngine() {
        //     return &eng_;
        // }

        // stream* getStream() {
        //     return &s_;
        // }

        cudnnHandle_t* getCudnnHandle() {
            return cudnnHandle_;
        }

        cublasHandle_t* getCublasHandle() {
            return cublasHandle_;
        }

        float* allocDevMem(size_t size) {
            float *d_ptr = (float *)sycl::malloc_device(size, *sycl_queue_);
            return d_ptr;
        }

        void freeDevPtr(float* devPtr) {
            sycl::free(devPtr, *sycl_queue_);   
        }

        void memCpy(float* devPtr, const float* hostPtr, size_t size, bool needToSynchronize, MemcpyType memcpyType) {
            sycl_queue_->memcpy(devPtr, hostPtr, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void memCpyH2D(float* devPtr, const float* hostPtr, size_t size, bool needToSynchronize) {
            sycl_queue_->memcpy(devPtr, hostPtr, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void memCpyD2D(float* devPtr1, const float* devPtr2, size_t size, bool needToSynchronize) {
            sycl_queue_->memcpy(devPtr1, devPtr2, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void memCpyD2H(float* hostPtr, const float* devPtr, size_t size, bool needToSynchronize) {
            sycl_queue_->memcpy(hostPtr, devPtr, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void synchronize() {
            sycl_queue_->wait();
        }

};

#elif defined(USE_ROCBLAS)
class LangHandle {
    private:
        Timer* timer_;
        miopenHandle_t *miopenHandle_;
        rocblas_handle *rocblasHandle_;

        sycl::device *dht_;
        sycl::context *context_;
        sycl::queue *sycl_queue_;  

    public:
        LangHandle(Timer* timer): timer_(timer) {
            //assertDevApiInvar(hipSetDevice(0));

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




            dht_ = new sycl::device(sycl::gpu_selector_v); 
#ifdef DEVICE_TIMER  
            start = get_time_now();
#endif    
            context_ = new sycl::context(*dht_);
#ifdef DEVICE_TIMER  
            timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_SYCL_CONTEXT");
#endif    
            auto propList = sycl::property_list{
                #ifdef IN_ORDER_QUEUE
                sycl::property::queue::in_order{},
                #ifdef SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
                sycl::ext::oneapi::property::queue::discard_events{}
                #endif
                #endif
            };
#ifdef DEVICE_TIMER  
            start = get_time_now();  
#endif     
            sycl_queue_ = new sycl::queue(*context_, *dht_, propList);
#ifdef DEVICE_TIMER  
            timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_SYCL_QUEUE");
#endif    

        }



        miopenHandle_t* getMiopenHandle() {
            return miopenHandle_;
        }

        rocblas_handle* getRocblasHandle() {
            return rocblasHandle_;
        }

        sycl::device* getDevice() {
            return dht_;
        }

        sycl::context* getContext() {
            return context_;
        }

        sycl::queue* getSyclQueue() {
            return sycl_queue_;
        }


        float* allocDevMem(size_t size) {
            float *d_ptr = (float *)sycl::malloc_device(size, *sycl_queue_);
            return d_ptr;
        }

        void freeDevPtr(float* devPtr) {
            sycl::free(devPtr, *sycl_queue_);   
        }

        void memCpy(float* devPtr, const float* hostPtr, size_t size, bool needToSynchronize, MemcpyType memcpyType) {
            sycl_queue_->memcpy(devPtr, hostPtr, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void memCpyH2D(float* devPtr, const float* hostPtr, size_t size, bool needToSynchronize) {
            sycl_queue_->memcpy(devPtr, hostPtr, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void memCpyD2D(float* devPtr1, const float* devPtr2, size_t size, bool needToSynchronize) {
            sycl_queue_->memcpy(devPtr1, devPtr2, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void memCpyD2H(float* hostPtr, const float* devPtr, size_t size, bool needToSynchronize) {
            sycl_queue_->memcpy(hostPtr, devPtr, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void synchronize() {
            sycl_queue_->wait();
        }

};

#else
class LangHandle {
    private:
        Timer* timer_;
        sycl::device *dht_;
        sycl::context *context_;
        sycl::queue *sycl_queue_;  
        engine eng_;   
        stream s_;

    public:
        LangHandle(Timer* timer): timer_(timer) {
            dht_ = new sycl::device(sycl::gpu_selector_v); 
#ifdef DEVICE_TIMER  
            start = get_time_now();
#endif    
            context_ = new sycl::context(*dht_);
#ifdef DEVICE_TIMER  
            timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_SYCL_CONTEXT");
#endif    
            auto propList = sycl::property_list{
                #ifdef IN_ORDER_QUEUE
                sycl::property::queue::in_order{},
                #ifdef SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
                sycl::ext::oneapi::property::queue::discard_events{}
                #endif
                #endif
            };
#ifdef DEVICE_TIMER  
            start = get_time_now();  
#endif     
            sycl_queue_ = new sycl::queue(*context_, *dht_, propList);
#ifdef DEVICE_TIMER  
            timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_SYCL_QUEUE");
#endif    
#ifdef DEVICE_TIMER  
            start = get_time_now();
#endif    
            //engine eng(engine::kind::gpu, 0);
            eng_ = dnnl::sycl_interop::make_engine(*dht_, *context_);
#ifdef DEVICE_TIMER  
            timer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_ONEDNN_ENGINE");
#endif    
#ifdef DEVICE_TIMER      
            start = get_time_now();
#endif    
            //stream s(eng);
            s_ = dnnl::sycl_interop::make_stream(eng_, *sycl_queue_);
#ifdef DEVICE_TIMER      
            imer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "CREATE_ONEDNN STREAM");
#endif    
        }


        sycl::device* getDevice() {
            return dht_;
        }

        sycl::context* getContext() {
            return context_;
        }

        sycl::queue* getSyclQueue() {
            return sycl_queue_;
        }

        engine* getEngine() {
            return &eng_;
        }

        stream* getStream() {
            return &s_;
        }

        float* allocDevMem(size_t size) {
            float *d_ptr = (float *)sycl::malloc_device(size, *sycl_queue_);
            return d_ptr;
        }

        void freeDevPtr(float* devPtr) {
            sycl::free(devPtr, *sycl_queue_);   
        }

        void memCpy(float* devPtr, const float* hostPtr, size_t size, bool needToSynchronize, MemcpyType memcpyType) {
            sycl_queue_->memcpy(devPtr, hostPtr, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void memCpyH2D(float* devPtr, const float* hostPtr, size_t size, bool needToSynchronize) {
            sycl_queue_->memcpy(devPtr, hostPtr, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void memCpyD2D(float* devPtr1, const float* devPtr2, size_t size, bool needToSynchronize) {
            sycl_queue_->memcpy(devPtr1, devPtr2, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void memCpyD2H(float* hostPtr, const float* devPtr, size_t size, bool needToSynchronize) {
            sycl_queue_->memcpy(hostPtr, devPtr, size);
            if(needToSynchronize) {
                sycl_queue_->wait();
            }
        }

        void synchronize() {
            sycl_queue_->wait();
        }

};
#endif

#endif

