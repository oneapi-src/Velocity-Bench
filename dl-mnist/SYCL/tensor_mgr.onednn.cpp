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
#include <iostream>
#include <iterator>
#include <list>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "dnnl_sycl.hpp"
#endif

#include "tensor_mgr.onednn.h"
#include "../common/mnist.h"
#include "../common/utils.h"
#include "../common/timing.h"
#include "../common/workload_params.h"
#include "../common/tracer.h"
#include "../common/exec_policies.h" 


using namespace dnnl;
using namespace dl_infra::common;
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

namespace dl_infra {
    namespace common {
                //This function was originally taken from oneDNN examples (example_utils.hpp) and modified
        // Read from handle, write to memory
        inline void TensorMgr::write_to_dnnl_memory2(void *handle, dnnl::memory &mem) {
            Tracer::func_begin("TensorMgr::write_to_dnnl_memory2");  

            dnnl::engine eng = mem.get_engine();
            size_t size = mem.get_desc().get_size();

            if (!handle) throw std::runtime_error("handle is nullptr.");

        #ifdef DNNL_WITH_SYCL
            bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
                    && eng.get_kind() == dnnl::engine::kind::gpu);
            if (is_gpu_sycl) {
                auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
                if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
                    //cout << "Using buffer for copy" << std::endl;
                    auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
                    auto dst = buffer.get_access<::sycl::access::mode::write>();
                    uint8_t *dst_ptr = dst.get_pointer();
                    if (!dst_ptr)
                        throw std::runtime_error("get_pointer returned nullptr.");
                    for (size_t i = 0; i < size; ++i)
                        dst_ptr[i] = ((uint8_t *)handle)[i];
                } else {
                    //cout << "Using USM for copy" << std::endl;
                    assert(mkind == dnnl::sycl_interop::memory_kind::usm);
                    uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
                    if (!dst_ptr)
                        throw std::runtime_error("get_data_handle returned nullptr.");

                    auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                    sycl_queue.memcpy(dst_ptr, handle, size).wait();

                }
                Tracer::func_end("TensorMgr::write_to_dnnl_memory2");                
                 
                return;
            }
        #endif

            assert(!"not expected");
        }

        inline void TensorMgr::write_to_dnnl_memory3(void *src_ptr, void *dst_ptr, size_t size) {
            Tracer::func_begin("TensorMgr::write_to_dnnl_memory3");  

            //dnnl::engine eng = mem.get_engine();
            //size_t size = mem.get_desc().get_size();

            //if (!handle) throw std::runtime_error("handle is nullptr.");


                //cout << "Using USM for copy" << std::endl;
                //assert(mkind == dnnl::sycl_interop::memory_kind::usm);
                //uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
                //if (!dst_ptr)
                //        throw std::runtime_error("get_data_handle returned nullptr.");
                std::cout << "Before get_queue" << std::endl;
                auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng_));
                std::cout << "After get_queue" << std::endl;
                sycl_queue.memcpy(dst_ptr, src_ptr, size).wait();
                std::cout << "After memcpy" << std::endl;

                Tracer::func_end("TensorMgr::write_to_dnnl_memory3");                   
                 
                return;

            assert(!"not expected");
        }


    



        TensorBag* TensorMgr::setupTensorsForConvLayer(int conv_layer_index, 
            int* input_tensor_dims_, int* filter_tensor_dims_, int* output_tensor_dims_) 
        {
            Tracer::func_begin("TensorMgr::setupTensorsForConvLayer");   

            tensorBags.at(conv_layer_index) = new TensorBag(timer_);
            createTensorDescriptors(conv_layer_index, input_tensor_dims_, filter_tensor_dims_, output_tensor_dims_);
            createIOTensors(conv_layer_index, input_tensor_dims_, output_tensor_dims_);

            Tracer::func_end("TensorMgr::setupTensorsForConvLayer");  

            return  tensorBags.at(conv_layer_index);
        }

        TensorBag* TensorMgr::createAndSetupTensorBag(int conv_layer_index, 
            int* input_tensor_dims_, int* filter_tensor_dims_, int* output_tensor_dims_) 
        {
            Tracer::func_begin("TensorMgr::createAndSetupTensorBag");   

            tensorBags.at(conv_layer_index) = new TensorBag(timer_);
            createTensorDescriptors(conv_layer_index, input_tensor_dims_, filter_tensor_dims_, output_tensor_dims_);
            
            Tracer::func_end("TensorMgr::createAndSetupTensorBag");  

            return  tensorBags.at(conv_layer_index);
        }

        TensorMgr::TensorMgr(WorkloadParams* workloadParams, Timer* timer, Timer* dataFileReadTimer, int no_of_layers, engine eng)
        : workloadParams_(workloadParams), eng_(std::move(eng)), timer_(timer), dataFileReadTimer_(dataFileReadTimer), no_of_layers_(no_of_layers) {

            Tracer::func_begin("TensorMgr::TensorMgr"); 

            tensorBags.resize(no_of_layers);

             Tracer::func_end("TensorMgr::TensorMgr");         
        }

        void TensorMgr::createIOTensors(int conv_layer_index, int* input_tensor_dims_, int* output_tensor_dims_) {
            Tracer::func_begin("TensorMgr::createIOTensors");   

            Time start;
            
            TensorBag* tensorBag = tensorBags.at(conv_layer_index);

            int input_size  =  input_tensor_dims_[0] *  input_tensor_dims_[1] *  input_tensor_dims_[2] *  input_tensor_dims_[3];
            int output_size = output_tensor_dims_[0] * output_tensor_dims_[1] * output_tensor_dims_[2] * output_tensor_dims_[3];

            tensorBag->ooc_output_host_ptr_ = (float*)calloc(output_size, sizeof(float));

            auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng_));

#ifdef DEVICE_TIMER    
            start = get_time_now();
#endif            
            void * d_output = sycl::malloc_device(output_size*sizeof(float), sycl_queue);
            tensorBag->dst_mem_ = memory(
                    {{output_tensor_dims_[0], output_tensor_dims_[1], output_tensor_dims_[2], output_tensor_dims_[3]},
                    memory::data_type::f32, memory::format_tag::nchw}, eng_, d_output);
            tensorBag->dst_mem_.set_data_handle(d_output);
            // tensorBag->dst_mem_ = memory(
            //         {{output_tensor_dims_[0], output_tensor_dims_[1], output_tensor_dims_[2], output_tensor_dims_[3]},
            //         memory::data_type::f32, memory::format_tag::nchw}, eng_);
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "MEMALLOC_DST_DEV_MEM");
#endif           
            assert(d_output);

            initImage(tensorBag->ooc_output_host_ptr_, output_size);

            //std::cout<< "MEM_SIZE - sizeof(float): " << sizeof(float) << ", output_size: " << output_size << ", sizeof(float) * output_size: " << sizeof(float) * output_size << "\t\tLayer: " << conv_layer_index << std::endl;
            Tracer::mem_op("MEM_SIZE - sizeof(float): " + to_string(sizeof(float)) + "B, output_size: " + to_string(output_size) + ", sizeof(float) * output_size: " + to_string((((double)(sizeof(float) * output_size)/1000000)/1000)) + " GB" + "\t\tLayer: " + to_string(conv_layer_index));

            if(conv_layer_index == 0) {
                int input_size  =  input_tensor_dims_[0] *  input_tensor_dims_[1] *  input_tensor_dims_[2] *  input_tensor_dims_[3];

                Time m_tStart = get_time_now();
                tensorBag->ooc_input_host_ptr_ = readMnistDataFiles2(input_tensor_dims_[0]);
                dataFileReadTimer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(m_tStart), "MNIST_READ");

#ifdef DEVICE_TIMER    
                start = get_time_now();
#endif                  
                void * d_input = sycl::malloc_device(input_size*sizeof(float), sycl_queue);
                tensorBag->src_mem_ = memory(
                        {{input_tensor_dims_[0], input_tensor_dims_[1], input_tensor_dims_[2], input_tensor_dims_[3]}, 
                        memory::data_type::f32, memory::format_tag::nchw}, eng_, d_input);
                tensorBag->src_mem_.set_data_handle(d_input);       
                // tensorBag->src_mem_ = memory(
                //         {{input_tensor_dims_[0], input_tensor_dims_[1], input_tensor_dims_[2], input_tensor_dims_[3]}, 
                //         memory::data_type::f32, memory::format_tag::nchw}, eng_);                 
#ifdef DEVICE_TIMER    
                timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "MEMALLOC_INPUT_DEV_MEM"); 
#endif                  
                 assert(d_input);

#ifdef DEVICE_TIMER    
                start = get_time_now();
#endif                  
                write_to_dnnl_memory2(tensorBag->ooc_input_host_ptr_, tensorBag->src_mem_);
#ifdef DEVICE_TIMER    
                timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "USM_MEMCPY_INPUT_FROM_HOST_TO_DEVICE"); 
#endif                  
            }
            if(conv_layer_index != 0) {
                tensorBag->ooc_input_host_ptr_ = tensorBags.at(conv_layer_index-1)->ooc_output_host_ptr_;
                tensorBag->src_mem_            = tensorBags.at(conv_layer_index-1)->dst_mem_;
            }

            Tracer::func_end("TensorMgr::createIOTensors");    
        }

        void TensorMgr::createWeightsTensor(int conv_layer_index, int* filter_tensor_dims_) {
            Tracer::func_begin("TensorMgr::createWeightsTensor");   

            Time start;
            
            TensorBag* tensorBag = tensorBags.at(conv_layer_index);

            int filter_size = filter_tensor_dims_[0] * filter_tensor_dims_[1] * filter_tensor_dims_[2] * filter_tensor_dims_[3];
            
            tensorBag->ooc_filter_host_ptr_ = (float*)calloc(filter_size, sizeof(float));

            auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng_));

#ifdef DEVICE_TIMER    
            start = get_time_now();
#endif              
            tensorBag->weights_mem_ = memory(
                    {{filter_tensor_dims_[0], filter_tensor_dims_[1], filter_tensor_dims_[2], filter_tensor_dims_[3]}, 
                    memory::data_type::f32, memory::format_tag::oihw}, eng_);
#ifdef DEVICE_TIMER    
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "MEMALLOC_WEIGHTS_DEV_MEM");
#endif              

            initImage(tensorBag->ooc_filter_host_ptr_, filter_size);
            
            //std::cout<< "MEM_SIZE - sizeof(float): " << sizeof(float) << ", filter_size: " << filter_size << ", sizeof(float) * filter_size: " << sizeof(float) * filter_size << "\t\tLayer: " << conv_layer_index << std::endl;
            
            Tracer::mem_op("MEM_SIZE - sizeof(float): " + to_string(sizeof(float)) + "B, filter_size: " + to_string(filter_size) + ", sizeof(float) * filter_size: " + to_string((((double)(sizeof(float) * filter_size)/1000000)/1000)) + " GB" + "\t\tLayer: " + to_string(conv_layer_index));

#ifdef DEVICE_TIMER                
            start = get_time_now();
#endif              
            write_to_dnnl_memory2(tensorBag->ooc_filter_host_ptr_, tensorBag->weights_mem_);
#ifdef DEVICE_TIMER    
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "USM_MEMCPY_WEIGHTS_FROM_HOST_TO_DEVICE");
#endif             

            Tracer::func_end("TensorMgr::createWeightsTensor");    
        }

        void TensorMgr::createTensorDescriptors(int conv_layer_index, int* input_tensor_dims_, int* filter_tensor_dims_, int* output_tensor_dims_) {
            Tracer::func_begin("TensorMgr::createTensorDescriptors");  

            TensorBag* tensorBag = tensorBags.at(conv_layer_index);

            memory::format_tag formatTag = memory::format_tag::any;

            if(workloadParams_->getOneDnnConvPdMemFormat() == WorkloadParams::OneDnnConvPdMemFormat::ONEDNN_CONVPD_NCHW) {
                formatTag = memory::format_tag::nchw;
            } else if(workloadParams_->getOneDnnConvPdMemFormat() == WorkloadParams::OneDnnConvPdMemFormat::ONEDNN_CONVPD_ANY) {
                formatTag = memory::format_tag::any;
            }

#ifdef DEVICE_TIMER    
            Time start = get_time_now();
#endif              
            tensorBag->conv_src_md = memory::desc({input_tensor_dims_[0], input_tensor_dims_[1], input_tensor_dims_[2], input_tensor_dims_[3]}, memory::data_type::f32,
                    formatTag //any  // let convolution choose memory format
            );
            tensorBag->conv_weights_md = memory::desc(
                    {filter_tensor_dims_[0], filter_tensor_dims_[1], filter_tensor_dims_[2], filter_tensor_dims_[3]}, memory::data_type::f32,
                    formatTag //any   // let convolution choose memory format
            );
            tensorBag->conv_dst_md = memory::desc({output_tensor_dims_[0], output_tensor_dims_[1], output_tensor_dims_[2], output_tensor_dims_[3]}, memory::data_type::f32,
                    formatTag //any   // let convolution choose memory format
            );
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(conv_layer_index, calculate_op_time_taken(start), "ALL_MEM_DESC CREATION");
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
