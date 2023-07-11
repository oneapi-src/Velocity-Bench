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

#include <iostream>
#include <cudnn.h>
#include <iostream>
#include <iterator>
#include <string>
#include <list>
#include <vector>

#include "tensor_mgr.cudnn.h"
#include "error_handling.cudnn.h"
#include "../common/mnist.h"
#include "../common/utils.h"
#include "../common/timing.h"
#include "../common/tracer.h"
#include "../common/exec_policies.h" 
#include "../common/workload_params.h"


//using namespace std;
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

namespace dl_infra {
    namespace common {
        TensorBag* TensorMgr::setupTensorsForConvLayer(int conv_layer_index, 
            int* input_tensor_dims_, int* filter_tensor_dims_, int* output_tensor_dims_, int input_stride_dims_[], int output_stride_dims_[]) 
        {
            Tracer::func_begin("TensorMgr::setupTensorsForConvLayer");    
            
            tensorBags.at(conv_layer_index) = new TensorBag(timer_);
            createTensorDescriptors(conv_layer_index, input_tensor_dims_, filter_tensor_dims_, output_tensor_dims_, input_stride_dims_, output_stride_dims_);
            createIOTensors(conv_layer_index, input_tensor_dims_, output_tensor_dims_);

            Tracer::func_end("TensorMgr::setupTensorsForConvLayer");  

            return  tensorBags.at(conv_layer_index);
        }

        TensorBag* TensorMgr::createAndSetupTensorBag(int conv_layer_index, 
            int* input_tensor_dims_, int* filter_tensor_dims_, int* output_tensor_dims_, int input_stride_dims_[], int output_stride_dims_[]) 
        {
            Tracer::func_begin("TensorMgr::createAndsetupTensorBag");    
            
            tensorBags.at(conv_layer_index) = new TensorBag(timer_);
            createTensorDescriptors(conv_layer_index, input_tensor_dims_, filter_tensor_dims_, output_tensor_dims_, input_stride_dims_, output_stride_dims_);

            Tracer::func_end("TensorMgr::createAndsetupTensorBag");  

            return  tensorBags.at(conv_layer_index);
        }

        TensorMgr::TensorMgr(WorkloadParams *workloadParams, Timer* timer, Timer *dataFileReadTimer, int no_of_layers): workloadParams_(workloadParams), timer_(timer), dataFileReadTimer_(dataFileReadTimer), no_of_layers_(no_of_layers) {
            Tracer::func_begin("TensorMgr::TensorMgr");   
            
            tensorBags.resize(no_of_layers);
            //std::cout<< "TensorMgr::TensorMgr - tensorBags.size" << tensorBags.size() << std::endl;

            Tracer::func_end("TensorMgr::TensorMgr");  
        }

        void TensorMgr::createIOTensors(int conv_layer_index, int* input_tensor_dims_, int* output_tensor_dims_) {
            Tracer::func_begin("TensorMgr::createIOTensors");   

            Time start;
            
            TensorBag* tensorBag = tensorBags.at(conv_layer_index);

            int input_size  =  input_tensor_dims_[0] *  input_tensor_dims_[1] *  input_tensor_dims_[2] *  input_tensor_dims_[3];
            int output_size = output_tensor_dims_[0] * output_tensor_dims_[1] * output_tensor_dims_[2] * output_tensor_dims_[3];

            tensorBag->ooc_output_host_ptr_ = (float*)calloc(output_size, sizeof(float));

#ifdef DEVICE_TIMER    
            start = get_time_now();
#endif            
            cudaMalloc((void**)&(tensorBag->output_dev_ptr_), (output_size) * sizeof(float));
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "CUDAMALLOC_OUTPUT_DEV_PTR");
#endif
            initImage(tensorBag->ooc_output_host_ptr_, output_size);

            //std::cout<< "MEM_SIZE - sizeof(float): " << sizeof(float) << ", output_size: " << output_size << ", sizeof(float) * output_size: " << sizeof(float) * output_size << "\t\tLayer: " << conv_layer_index << std::endl;

            Tracer::mem_op("MEM_SIZE - sizeof(float): " + to_string(sizeof(float)) + "B, output_size: " + to_string(output_size) + ", sizeof(float) * output_size: " + to_string((((double)(sizeof(float) * output_size)/1000000)/1000)) + " GB" + "\t\tLayer: " + to_string(conv_layer_index));

            if(conv_layer_index == 0) {
                int input_size  =  input_tensor_dims_[0] *  input_tensor_dims_[1] *  input_tensor_dims_[2] *  input_tensor_dims_[3];

                Time m_tStart = get_time_now();
                tensorBag->ooc_input_host_ptr_ = readMnistDataFiles2(input_tensor_dims_[0]);
                dataFileReadTimer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(m_tStart), "MNIST_READ");
                //auto m_tStop = std::chrono::steady_clock::now();
                //std::cout <<"Datafile read time: " << std::chrono::duration<double>(m_tStop - m_tStart).count() << " sec." << std::endl;


#ifdef DEVICE_TIMER    
                Time start = get_time_now();
#endif                
                cudaMalloc((void**)&(tensorBag->input_dev_ptr_), (input_size) * sizeof(float));
                //if(!dryRun_) { timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "CUDAMALLOC_INPUT_DEV_PTR"); }
#ifdef DEVICE_TIMER                    
                timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "CUDAMALLOC_INPUT_DEV_PTR");
#endif                
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                assertCudaInvar(cudaMemcpy(tensorBag->input_dev_ptr_,  tensorBag->ooc_input_host_ptr_,  sizeof(float) * input_size, cudaMemcpyHostToDevice));
                assertCudaInvar(cudaDeviceSynchronize());
                //if(!dryRun_) { timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "CUDAMEMCPY_INPUT_HOST_TO_DEV"); }
#ifdef DEVICE_TIMER                    
                timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "CUDAMEMCPY_INPUT_HOST_TO_DEV");
#endif
            }
            if(conv_layer_index != 0) {
                tensorBag->ooc_input_host_ptr_ = tensorBags.at(conv_layer_index-1)->ooc_output_host_ptr_;
                tensorBag->input_dev_ptr_      = tensorBags.at(conv_layer_index-1)->output_dev_ptr_;
            }

            Tracer::func_end("TensorMgr::createIOTensors");         
        }

        void TensorMgr::createWeightsTensor(int conv_layer_index, int* filter_tensor_dims_) {
            Tracer::func_begin("TensorMgr::createWeightsTensor");   

            Time start;
            
            TensorBag* tensorBag = tensorBags.at(conv_layer_index);

            int filter_size = filter_tensor_dims_[0] * filter_tensor_dims_[1] * filter_tensor_dims_[2] * filter_tensor_dims_[3];
            
            tensorBag->ooc_filter_host_ptr_ = (float*)calloc(filter_size, sizeof(float));
           
#ifdef DEVICE_TIMER               
            start = get_time_now();
#endif            
            cudaMalloc((void**)&(tensorBag->filter_dev_ptr_), (filter_size) * sizeof(float));
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "CUDAMALLOC_FILTER_DEV_PTR");
#endif
            
            initImage(tensorBag->ooc_filter_host_ptr_, filter_size);
            
            //std::cout<< "MEM_SIZE - sizeof(float): " << sizeof(float) << ", filter_size: " << filter_size << ", sizeof(float) * filter_size: " << sizeof(float) * filter_size << "\t\tLayer: " << conv_layer_index << std::endl;
            Tracer::mem_op("MEM_SIZE - sizeof(float): " + to_string(sizeof(float)) + "B, filter_size: " + to_string(filter_size) + ", sizeof(float) * filter_size: " + to_string((((double)(sizeof(float) * filter_size)/1000000)/1000)) + " GB" + "\t\tLayer: " + to_string(conv_layer_index));

#ifdef DEVICE_TIMER                
            start = get_time_now();
#endif            
            assertCudaInvar(cudaMemcpy(tensorBag->filter_dev_ptr_, tensorBag->ooc_filter_host_ptr_, sizeof(float) * filter_size,    cudaMemcpyHostToDevice));
            assertCudaInvar(cudaDeviceSynchronize());
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "CUDAMEMCPY_FILTER_HOST_TO_DEV");
#endif            
            
            Tracer::func_end("TensorMgr::createWeightsTensor");         
        }

        void TensorMgr::createTensorDescriptors(int conv_layer_index, int* input_tensor_dims_, int* filter_tensor_dims_, int* output_tensor_dims_, int input_stride_dims_[], int output_stride_dims_[]) {
            Tracer::func_begin("TensorMgr::createTensorDescriptors");   

            TensorBag* tensorBag = tensorBags.at(conv_layer_index);
            
#ifdef DEVICE_TIMER                
            Time start = get_time_now();
#endif            
            assertCudnnInvar(cudnnCreateTensorDescriptor(&tensorBag->cudnn_input_desc_));
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "CREATE_INPUT_TENSOR_DESCRIPTOR");
#endif
#ifdef DEVICE_TIMER                
            start = get_time_now();
#endif            
            assertCudnnInvar(cudnnSetTensorNdDescriptor(tensorBag->cudnn_input_desc_, CUDNN_DATA_FLOAT, 4, input_tensor_dims_, input_stride_dims_));
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "SET_INPUT_TENSOR_ND_DESCRIPTOR");
#endif
#ifdef DEVICE_TIMER    
            start = get_time_now();
#endif            
            assertCudnnInvar(cudnnCreateFilterDescriptor(&tensorBag->cudnn_filter_desc_));
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "CREATE_FILTER_TENSOR_DESCRIPTOR");
#endif
#ifdef DEVICE_TIMER                
            start = get_time_now();
#endif            
            assertCudnnInvar(cudnnSetFilterNdDescriptor(tensorBag->cudnn_filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filter_tensor_dims_));
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "SET_FILTER_TENSOR_ND_DESCRIPTOR");
#endif
#ifdef DEVICE_TIMER    
            start = get_time_now();
#endif            
            assertCudnnInvar(cudnnCreateTensorDescriptor(&tensorBag->cudnn_output_desc_));
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "CREATE_OUTPUT_TENSOR_DESCRIPTOR");
#endif
#ifdef DEVICE_TIMER    
            start = get_time_now();
#endif            
            assertCudnnInvar(cudnnSetTensorNdDescriptor(tensorBag->cudnn_output_desc_, CUDNN_DATA_FLOAT, 4, output_tensor_dims_, output_stride_dims_));
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "SET_OUTPUT_TENSOR_ND_DESCRIPTOR");
#endif
            Tracer::func_end("TensorMgr::createTensorDescriptors");  
       }
       

        TensorMgr::~TensorMgr() {
            Tracer::func_begin("TensorMgr::~TensorMgr");   

            for(auto tensorBag: tensorBags) {
                delete tensorBag;
            }

            Tracer::func_end("TensorMgr::~TensorMgr");             
        }


        TensorBag::TensorBag(Timer* timer): timer_(timer) {
            cudnn_input_desc_  = 0;
            cudnn_filter_desc_ = 0;
            cudnn_output_desc_ = 0;

            workSpace_     = 0;
            workSpaceSize_ = 0;
        }


        TensorBag::~TensorBag() {
            Tracer::func_begin("TensorBag::~TensorBag");   

            if (ooc_input_host_ptr_) {
                   //free(input_host_ptr_);
                   delete ooc_input_host_ptr_;
                   ooc_input_host_ptr_ = NULL;
            }
            if (ooc_filter_host_ptr_) {
                  free(ooc_filter_host_ptr_);
                  ooc_filter_host_ptr_ = NULL;
            }
            if (ooc_output_host_ptr_) {
                  free(ooc_output_host_ptr_);
                  ooc_output_host_ptr_ = NULL;
            }

            if (cudnn_input_desc_) {
                 cudnnDestroyTensorDescriptor(cudnn_input_desc_);
                 cudnn_input_desc_ = NULL;
            }
            if (cudnn_filter_desc_) {
                cudnnDestroyFilterDescriptor(cudnn_filter_desc_);
                cudnn_filter_desc_ = NULL;
            }
            if (cudnn_output_desc_) {
                cudnnDestroyTensorDescriptor(cudnn_output_desc_);
                cudnn_output_desc_ = NULL;
            }

            Tracer::func_end("TensorBag::~TensorBag");              
        }

        float* TensorMgr::readMnistDataFiles2(int no_of_images) {                
            Tracer::func_begin("TensorMgr::readMnistDataFiles2");   

            string mnist_dataset_dir = "../../datasets/";
            int total_dataset_images_count, image_size;
            Exec_setup exec_setup = mnist_paper_1_exec_setup;
            //float* mnist_images_float2 = NULL;

            Tracer::func_end("TensorMgr::readMnistDataFiles2");  

            return read_mnist_images_float2(no_of_images, mnist_dataset_dir + "train-images.idx3-ubyte", total_dataset_images_count, image_size);
            
        }
    }
}