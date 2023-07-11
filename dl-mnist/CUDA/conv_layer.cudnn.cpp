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
#include <chrono>
#include <cudnn.h>

#include "error_handling.cudnn.h"
#include "conv_layer.cudnn.h"
#include "../common/mnist.h"
#include "../common/utils.h"
#include "../common/timing.h"
#include "../common/tracer.h"
#include "../common/exec_policies.h" 
#include "../common/workload_params.h"

#if defined(__linux__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#else
#error unsupported platform
#endif

using namespace dl_infra::common;
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

namespace dl_infra {
    namespace cudnn {

        ConvLayer::ConvLayer(WorkloadParams* workloadParams, int index_in_network, int total_layers_in_nw, Timer* timer, TensorMgr* tensor_mgr, cudnnHandle_t handle,  
                    int input_tensor_dims[], int filter_tensor_dims[], int output_tensor_dims[]): workloadParams_(workloadParams) {

            Tracer::func_begin("ConvLayer::ConvLayer");

            index_in_network_ = index_in_network;
            total_layers_in_nw_ = total_layers_in_nw;
            timer_ = timer;
            handle_ = handle;
            tensor_mgr_ = tensor_mgr;

            input_tensor_dims_  = input_tensor_dims;
            filter_tensor_dims_ = filter_tensor_dims;
            output_tensor_dims_ = output_tensor_dims;

            workSpace_ = 0;
            workSpaceSize_ = 0;

            cudnn_conv_desc_ = 0;

            Tracer::func_end("ConvLayer::ConvLayer");         
        };

    


        void ConvLayer::initialize() {
            Tracer::func_begin("ConvLayer::initialize");
            Time start;

            calculateStrideDims();
            tensor_mgr_->createAndSetupTensorBag(index_in_network_, input_tensor_dims_, filter_tensor_dims_, output_tensor_dims_, input_stride_dims_, output_stride_dims_);
            tensor_mgr_->createWeightsTensor(index_in_network_, filter_tensor_dims_);
#ifdef DEVICE_TIMER    
            start = get_time_now();
#endif            
            assertCudnnInvar(cudnnCreateConvolutionDescriptor(&cudnn_conv_desc_));
#ifdef DEVICE_TIMER        
            timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CREATE_CONVOLUTION_DECSRIPTOR");
#endif
#ifdef DEVICE_TIMER    
            start = get_time_now();
#endif            
            assertCudnnInvar(cudnnSetConvolutionMathType(cudnn_conv_desc_, CUDNN_FMA_MATH));
#ifdef DEVICE_TIMER        
            timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "SET_CONVOLUTION_MATH_TYPE");
#endif
#ifdef DEVICE_TIMER    
            start = get_time_now();
#endif            
            assertCudnnInvar(cudnnSetConvolutionNdDescriptor(cudnn_conv_desc_, 2, pad_, conv_stride_, dilation_, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#ifdef DEVICE_TIMER        
            timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "SET_CONVOLUTION_ND_DESCRIPTOR");
#endif

            if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::CUDNN_IMPLICIT_GEMM) {
                algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            } else if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::CUDNN_IMPLICIT_PRECOMP_GEMM) {
                algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            } else if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::CUDNN_GEMM) {
                algo_ = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
            } else if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::CUDNN_DIRECT) {
                algo_ = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
            } else if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::CUDNN_FFT) {
                algo_ = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
            } else if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::CUDNN_FFT_TILING) {
                algo_ = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
            } else if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::CUDNN_WINOGRAD) {
                algo_ = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
            } else if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::CUDNN_WINOGRAD_NONFUSED) {
                algo_ = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
            } else if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::CUDNN_FIND_BEST_ALGO) {
                findBestAlgo();
            }

            createWorkspace();
            
            Tracer::func_end("ConvLayer::initialize");  
        }

        void ConvLayer::findBestAlgo() {
            int returnedAlgoCount;
            cudnnConvolutionFwdAlgoPerf_t     perfResults;
#ifdef DEVICE_TIMER    
            Time start = get_time_now();
#endif            
            assertCudnnInvar(cudnnFindConvolutionForwardAlgorithm(handle_,
                                            tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_input_desc_,
                                            tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_filter_desc_,
                                            cudnn_conv_desc_,
                                            tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_output_desc_,
                                            1,
                                            &returnedAlgoCount,
                                            &perfResults));
#ifdef DEVICE_TIMER        
            timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "FIND CONVOLUTION FORWARD ALGORITHM");
#endif

            //cout << "returnedAlgoCount: " << returnedAlgoCount << endl;
            //cout << "perfResults:" << perfResults.algo << ", status: " << perfResults.status << endl;

            assert(!perfResults.status);
            algo_ = perfResults.algo;
        }

        void ConvLayer::doIOTensorAndWSAllocs() {
            Tracer::func_begin("ConvLayer::doIOTensorAndWSAllocs");

            tensor_mgr_->createIOTensors(index_in_network_, input_tensor_dims_, output_tensor_dims_);
            
            Tracer::func_end("ConvLayer::doIOTensorAndWSAllocs");  
        }

        void ConvLayer::doTensorAndWSDeallocs() {
            Time start;

            if (tensor_mgr_->getTensorBagAt(index_in_network_)->input_dev_ptr_) {
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                cudaFree(tensor_mgr_->getTensorBagAt(index_in_network_)->input_dev_ptr_);
#ifdef DEVICE_TIMER        
                timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CUDAFREE_INPUT_DEV_PTR");
#endif
                tensor_mgr_->getTensorBagAt(index_in_network_)->input_dev_ptr_ = NULL;
            }

            if(index_in_network_ == (total_layers_in_nw_-1)) {
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                cudaFree(tensor_mgr_->getTensorBagAt(index_in_network_)->output_dev_ptr_);
#ifdef DEVICE_TIMER        
                timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CUDAFREE_OUTPUT_DEV_PTR");
#endif
            }

            
        }

        void ConvLayer::stage1Cleanup() {
            Time start;

            if (tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_input_desc_) {
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                cudnnDestroyTensorDescriptor(tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_input_desc_);
#ifdef DEVICE_TIMER        
                timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "DESTROY_INPUT_DESCRIPTOR");
#endif
                tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_input_desc_ = NULL;
            }
            if (tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_filter_desc_) {
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                cudnnDestroyFilterDescriptor(tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_filter_desc_);
#ifdef DEVICE_TIMER        
                timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "DESTROY_FILTER_DESCRIPTOR");
#endif
                tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_filter_desc_ = NULL;
            }
            if (tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_output_desc_) {
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                cudnnDestroyTensorDescriptor(tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_output_desc_);
#ifdef DEVICE_TIMER        
                timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "DESTROY_OUTPUT_DESCRIPTOR");
#endif
                tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_output_desc_ = NULL;
            }
            if (cudnn_conv_desc_) {
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                cudnnDestroyConvolutionDescriptor(cudnn_conv_desc_);            
#ifdef DEVICE_TIMER        
                timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "DESTROY_CONVOUTION_DESCRIPTOR");
#endif
                cudnn_conv_desc_ = NULL;
            }

            if (workSpace_) {
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                cudaFree(workSpace_);
#ifdef DEVICE_TIMER        
                timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CUDAFREE_WORKSPACE");
#endif
                workSpace_ = NULL;
            }
        }


        void ConvLayer::createWorkspace() {
            Tracer::func_begin("ConvLayer::createWorkspace"); 
            
            Time start;
            workSpace_ = 0;

            size_t free, total;
            int id;
            cudaGetDevice( &id );
            cudaMemGetInfo( &free, &total );
            //cout << "GPU " << id << " memory: free=" << (((double)free/1000000)/1000) << ", total=" << (((double)total/1000000)/1000) << endl;

#ifdef DEVICE_TIMER                
            start = get_time_now();
#endif            
            assertCudnnInvar(cudnnGetConvolutionForwardWorkspaceSize(
            handle_, tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_input_desc_, tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_filter_desc_, cudnn_conv_desc_, tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_output_desc_, algo_, &workSpaceSize_));
#ifdef DEVICE_TIMER        
            timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE");
#endif
            Tracer::mem_op("MEM_SIZE - required workpace size before allocation: " + to_string((((double)workSpaceSize_/1000000)/1000)) + " GB" + "\t\t\t\t\tLayer: " + to_string(index_in_network_));
            //std::cout<< "MEM_SIZE - required workpace size before allocation: " << (((double)workSpaceSize_/1000000)/1000) << " GB" << "\t\t\t\t\tLayer: " << index_in_network_ << std::endl;
            
            if (workSpaceSize_ > 0) {
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                assertCudaInvar(cudaMalloc(&workSpace_, workSpaceSize_));
#ifdef DEVICE_TIMER                        
                timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CUDAMALLOC_WORKSPACE");
#endif
            }

            Tracer::func_end("ConvLayer::createWorkspace");  
        };


        void ConvLayer::doConv(double prev_cum_time_taken) {

            Tracer::func_begin("ConvLayer::doConv");

            // if(initialized_ == false) {
            //     //TODO: throw exception instead
            //     return; 
            // }

            totalTime += prev_cum_time_taken;
#ifdef DEVICE_TIMER                
            Time start = get_time_now();
#endif            
            assertCudnnInvar(cudnnConvolutionForward(handle_,
                                          (void*)(&alpha_),
                                          tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_input_desc_,
                                          tensor_mgr_->getTensorBagAt(index_in_network_)->input_dev_ptr_,
                                          tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_filter_desc_,
                                          tensor_mgr_->getTensorBagAt(index_in_network_)->filter_dev_ptr_,
                                          cudnn_conv_desc_,
                                          algo_,
                                          workSpace_,
                                          workSpaceSize_,
                                          (void*)(&beta_),
                                          tensor_mgr_->getTensorBagAt(index_in_network_)->cudnn_output_desc_,
                                          tensor_mgr_->getTensorBagAt(index_in_network_)->output_dev_ptr_));
            assertCudaInvar(cudaDeviceSynchronize());
#ifdef DEVICE_TIMER                    
            timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CONV_FORWARD");
#endif
            Tracer::func_end("ConvLayer::doConv");  
        };


        

        void ConvLayer::printStatus() {
            //std::cout << "End of conv layer processing" << std::endl;
        };


        void ConvLayer::calculateStrideDims() {
            Tracer::func_begin("ConvLayer::calculateStrideDims");        

            //int input_stride_dims[] = {784, 784, 28, 1};
            //int output_stride_dims[] = {21632, 676, 26, 1};

            //TODO: this needs to change to a more concise implementation
            //for now we have a very simplistic impl  
            input_stride_dims_[3] = 1;
            input_stride_dims_[2] = input_tensor_dims_[3];
            input_stride_dims_[1] = input_tensor_dims_[2] * input_tensor_dims_[3];
            input_stride_dims_[0] = input_tensor_dims_[1] * input_tensor_dims_[2] * input_tensor_dims_[3];

            output_stride_dims_[3] = 1;
            output_stride_dims_[2] = output_tensor_dims_[3];
            output_stride_dims_[1] = output_tensor_dims_[2] * output_tensor_dims_[3];
            output_stride_dims_[0] = output_tensor_dims_[1] * output_tensor_dims_[2] * output_tensor_dims_[3];

            Tracer::func_end("ConvLayer::calculateStrideDims");               
        }
        

        ConvLayer::~ConvLayer() {
            Tracer::func_begin("ConvLayer::~ConvLayer");    

            Time start;
            
            // if (input_dev_ptr_) {
            //     start = get_time_now();
            //     cudaFree(input_dev_ptr_);
            //     timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CUDAFREE_INPUT_DEV_PTR");
            //     input_dev_ptr_ = NULL;
            // }
            // if (filter_dev_ptr_) {
            //     start = get_time_now();
            //     cudaFree(filter_dev_ptr_);
            //     timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CUDAFREE_FILTER_DEV_PTR");
            //     filter_dev_ptr_ = NULL;
            // }
            // if (output_dev_ptr_) {
            //     start = get_time_now();
            //     cudaFree(output_dev_ptr_);
            //     timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CUDAFREE_OUTPUT_DEV_PTR");
            //     output_dev_ptr_ = NULL;
            // }
            if (workSpace_) {
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                cudaFree(workSpace_);
                //if(!dryRun_) { timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CUDAFREE_WORKSPACE"); }
#ifdef DEVICE_TIMER                        
                timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CUDAFREE_WORKSPACE");
#endif                
                workSpace_ = NULL;
            }

            
            // if (input_host_ptr_) {
            //        //free(input_host_ptr_);
            //        delete input_host_ptr_;
            //        input_host_ptr_ = NULL;
            // }
            // if (filter_host_ptr_) {
            //       free(filter_host_ptr_);
            //       filter_host_ptr_ = NULL;
            // }
            // if (output_host_ptr_) {
            //       free(output_host_ptr_);
            //       output_host_ptr_ = NULL;
            // }
            // if (cudnn_input_desc_) {
            //      cudnnDestroyTensorDescriptor(cudnn_input_desc_);
            //      cudnn_input_desc_ = NULL;
            // }
            // if (cudnn_filter_desc_) {
            //     cudnnDestroyFilterDescriptor(cudnn_filter_desc_);
            //     cudnn_filter_desc_ = NULL;
            // }
            // if (cudnn_output_desc_) {
            //     cudnnDestroyTensorDescriptor(cudnn_output_desc_);
            //     cudnn_output_desc_ = NULL;
            // }
            if (cudnn_conv_desc_) {
                cudnnDestroyConvolutionDescriptor(cudnn_conv_desc_);            
                cudnn_conv_desc_ = NULL;
            }

            Tracer::func_end("ConvLayer::~ConvLayer");                     
        }
    }
}


