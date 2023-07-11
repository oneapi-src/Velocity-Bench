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

#include "oneapi/dnnl/dnnl.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "dnnl_sycl.hpp"
#endif

#include "conv_layer.onednn.h"
#include "../common/mnist.h"
#include "../common/utils.h"
#include "../common/timing.h"
#include "../common/tracer.h"
#include "../common/exec_policies.h" 

using namespace dnnl;
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;


namespace dl_infra {
    namespace onednn {


        //This function was originally taken from oneDNN examples (example_utils.hpp) and modified
        // Read from handle, write to memory
        inline void ConvLayer::write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
            Tracer::func_begin("ConvLayer::write_to_dnnl_memory");

            dnnl::engine eng = mem.get_engine();
            size_t size = mem.get_desc().get_size();

            if (!handle) throw std::runtime_error("handle is nullptr.");

        #ifdef DNNL_WITH_SYCL
            bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
                    && eng.get_kind() == dnnl::engine::kind::gpu);
            if (is_gpu_sycl) {
                auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
                if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
                    cout << "Using buffer for copy" << std::endl;
                    auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
                    auto dst = buffer.get_access<::sycl::access::mode::write>();
                    uint8_t *dst_ptr = dst.get_pointer();
                    if (!dst_ptr)
                        throw std::runtime_error("get_pointer returned nullptr.");
                    for (size_t i = 0; i < size; ++i)
                        dst_ptr[i] = ((uint8_t *)handle)[i];
                } else {
                    cout << "Using USM for copy" << std::endl;
                    assert(mkind == dnnl::sycl_interop::memory_kind::usm);
                    uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
                    if (!dst_ptr)
                        throw std::runtime_error("get_data_handle returned nullptr.");

                    auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                    sycl_queue.memcpy(dst_ptr, handle, size).wait();

                }
                Tracer::func_end("ConvLayer::write_to_dnnl_memory");                            
                return;
            }
        #endif

            assert(!"not expected");
  
        }

        ConvLayer::ConvLayer(WorkloadParams* workloadParams, int index_in_network, int total_layers_in_nw, 
            Timer* timer, TensorMgr* tensor_mgr, engine eng, stream s, 
            int input_tensor_dims[], int filter_tensor_dims[], int output_tensor_dims[]): workloadParams_(workloadParams)  {
            
            Tracer::func_begin("ConvLayer::ConvLayer");

            index_in_network_ = index_in_network;
            total_layers_in_nw_ = total_layers_in_nw;
            timer_ = timer;
            eng_ = std::move(eng);
            s_ = std::move(s);
            tensor_mgr_ = tensor_mgr;

            input_tensor_dims_  = input_tensor_dims;
            filter_tensor_dims_ = filter_tensor_dims;
            output_tensor_dims_ = output_tensor_dims;

            Tracer::func_end("ConvLayer::ConvLayer");    
        }

        ConvLayer::ConvLayer(WorkloadParams* workloadParams, int index_in_network, int total_layers_in_nw, 
            Timer* timer, TensorMgr* tensor_mgr, IConvLayer* nextConvLayer, engine eng, stream s, 
            int input_tensor_dims[], int filter_tensor_dims[], int output_tensor_dims[])
                    : ConvLayer(workloadParams, index_in_network, total_layers_in_nw, timer, tensor_mgr, std::move(eng), std::move(s), input_tensor_dims, filter_tensor_dims, output_tensor_dims) {
            nextConvLayer_ = nextConvLayer;                        
        };


        void ConvLayer::initialize() {
            Tracer::func_begin("ConvLayer::initialize");

            calculateStrideDims();
            tensor_mgr_->createAndSetupTensorBag(index_in_network_, input_tensor_dims_, filter_tensor_dims_, output_tensor_dims_);
            tensor_mgr_->createWeightsTensor(index_in_network_, filter_tensor_dims_);

            algorithm algo = algorithm::convolution_auto;
            if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::ONEDNN_AUTO) {
                algo = algorithm::convolution_auto;
            } else if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::ONEDNN_DIRECT) {
                algo = algorithm::convolution_direct;
            } else if(workloadParams_->getConvAlgo() == WorkloadParams::ConvAlgo::ONEDNN_WINOGRAD) {
                algo = algorithm::convolution_winograd;
            }  

#ifdef DEVICE_TIMER    
            Time start = get_time_now();
#endif            
            conv_pd = convolution_forward::primitive_desc(eng_,
            prop_kind::forward_inference, algo,
                    tensor_mgr_->getTensorBagAt(index_in_network_)->conv_src_md, 
                    tensor_mgr_->getTensorBagAt(index_in_network_)->conv_weights_md,
                    tensor_mgr_->getTensorBagAt(index_in_network_)->conv_dst_md, 
                    {1, 1}, // strides
                    {0, 0}, {0, 0} // left and right padding
            );
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CONV_PD CREATION");
#endif
#ifdef DEVICE_TIMER    
            start = get_time_now();
#endif            
            conv_ = convolution_forward(conv_pd);
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CONV_FORWARD CREATION");
#endif
            createWorkspace();

            Tracer::func_end("ConvLayer::initialize");  
        }

        void ConvLayer::doIOTensorAndWSAllocs() {
            Tracer::func_begin("ConvLayer::doTensorAndWSAllocs");

            tensor_mgr_->createIOTensors(index_in_network_, input_tensor_dims_, output_tensor_dims_);

            Tracer::func_end("ConvLayer::doTensorAndWSAllocs");  
        }

        void ConvLayer::doTensorAndWSDeallocs() {
#ifdef DEVICE_TIMER                
            Time start = get_time_now();
#endif            
            auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng_));
            sycl::free(tensor_mgr_->getTensorBagAt(index_in_network_)->src_mem_.get_data_handle(), sycl_queue);
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "FREE_INPUT_DEV_PTR");
#endif            
            //tensor_mgr_->getTensorBagAt(index_in_network_)->src_mem_.set_data_handle(NULL);

            if(index_in_network_ == (total_layers_in_nw_-1)) {
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                sycl::free(tensor_mgr_->getTensorBagAt(index_in_network_)->dst_mem_.get_data_handle(), sycl_queue);
#ifdef DEVICE_TIMER                    
                timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "FREE_OUTPUT_DEV_PTR");
#endif                
            }
        }
        void ConvLayer::stage1Cleanup() {}
        void ConvLayer::findBestAlgo() {}

        void ConvLayer::createWorkspace() {
            Tracer::func_begin("ConvLayer::createWorkspace");

            double elapsedTime;
#ifdef DEVICE_TIMER                
            Time start = get_time_now();
#endif            
            conv_scratchpad_mem_ = memory(conv_pd.scratchpad_desc(), eng_);
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "MEMALLOC_SCRATCHPAD_DEV_MEM");
#endif
            Tracer::func_end("ConvLayer::createWorkspace");    
        };

        void ConvLayer::calculateStrideDims() {
            Tracer::func_begin("ConvLayer::calculateStrideDims");


            Tracer::func_end("ConvLayer::calculateStrideDims");    
        }

        void ConvLayer::doConv(double prev_cum_time_taken) {
            Tracer::func_begin("ConvLayer::doConv");

            Time start;

        //     if(initialized_ == false) {
        //         //TODO: throw exception instead
        //         return; 
        //     }

            totalTime += prev_cum_time_taken;

            double elapsedTime;

            need_reorder_src_     = conv_pd.src_desc()     != tensor_mgr_->getTensorBagAt(index_in_network_)->src_mem_.get_desc();
            
            //need_reorder_weights_ = conv_pd.weights_desc() != tensor_mgr_->getTensorBagAt(index_in_network_)->weights_mem_.get_desc();
            
            if(index_in_network_ == total_layers_in_nw_-1) {
                need_reorder_dst_     = conv_pd.dst_desc()     != tensor_mgr_->getTensorBagAt(index_in_network_)->dst_mem_.get_desc();
            }
            
            //cout << "need_reorder_src: "     << need_reorder_src_ << endl;
            //cout << conv_pd.dst_desc().get_format_kind()
            //cout << "need_reorder_weights: " << need_reorder_weights_ << endl;
            //cout << "need_reorder_dst: "     << need_reorder_dst_ << endl;

            //if(index_in_network_ ==0) {
#ifdef DEVICE_TIMER                    
                start = get_time_now();
#endif                
                auto conv_src_mem     = need_reorder_src_ ? memory(conv_pd.src_desc(), eng_) : tensor_mgr_->getTensorBagAt(index_in_network_)->src_mem_;
                //auto conv_weights_mem = need_reorder_weights_ ? memory(conv_pd.weights_desc(), eng_) : tensor_mgr_->getTensorBagAt(index_in_network_)->weights_mem_;
                
                // in this workload we will forego reordering of weights
                // we will assume that the pre-trained weights have been created in the memory format as determined by conv_pd.weights_desc()
                auto conv_weights_mem = tensor_mgr_->getTensorBagAt(index_in_network_)->weights_mem_;     
                auto conv_dst_mem     = memory(conv_pd.dst_desc(), eng_, tensor_mgr_->getTensorBagAt(index_in_network_)->dst_mem_.get_data_handle());
                tensor_mgr_->getTensorBagAt(index_in_network_)->dst_mem_ = conv_dst_mem;
#ifdef DEVICE_TIMER                    
                timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "REORDERED MEM CREATE");
#endif                

                if (need_reorder_src_) {
#ifdef DEVICE_TIMER                        
                    start = get_time_now();
#endif                    
                    auto reorder_src = reorder(tensor_mgr_->getTensorBagAt(index_in_network_)->src_mem_, conv_src_mem);
                    reorder_src.execute(
                            s_, {{DNNL_ARG_FROM, tensor_mgr_->getTensorBagAt(index_in_network_)->src_mem_}, {DNNL_ARG_TO, conv_src_mem}});
                    s_.wait(); // wait for the reorder to complete
#ifdef DEVICE_TIMER                        
                    timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "REORDER SRC");
#endif                    
                }

                // if (need_reorder_weights_) {
                //     //start = get_time_now();
                //     auto reorder_weights = reorder(tensor_mgr_->getTensorBagAt(index_in_network_)->weights_mem_, conv_weights_mem);
                //     reorder_weights.execute(s_,
                //             {{DNNL_ARG_FROM, tensor_mgr_->getTensorBagAt(index_in_network_)->weights_mem_},
                //                     {DNNL_ARG_TO, conv_weights_mem}});
                //     s_.wait(); // wait for the reorder to complete
                //     timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "REORDER WEIGHTS"); 
                // }
            //}

#ifdef DEVICE_TIMER    
            start = get_time_now();
#endif            
            // conv_.execute(s_,
            //     {{DNNL_ARG_SRC, tensor_mgr_->getTensorBagAt(index_in_network_)->src_mem_}, {DNNL_ARG_WEIGHTS, tensor_mgr_->getTensorBagAt(index_in_network_)->weights_mem_},
            //         {DNNL_ARG_DST, tensor_mgr_->getTensorBagAt(index_in_network_)->dst_mem_}});
            conv_.execute(s_,
               {{DNNL_ARG_SRC, conv_src_mem}, {DNNL_ARG_WEIGHTS, conv_weights_mem},
                       {DNNL_ARG_DST, conv_dst_mem}});            
            s_.wait();
#ifdef DEVICE_TIMER                
            timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "CONV_FORWARD EXECUTION");
#endif
            // if(index_in_network_ == total_layers_in_nw_-1) {
            //     if (need_reorder_dst_) {
            //         start = get_time_now();
            //         auto reorder_dst = reorder(conv_dst_mem, tensor_mgr_->getTensorBagAt(index_in_network_)->dst_mem_);
            //         reorder_dst.execute(
            //                 s_, {{DNNL_ARG_FROM, conv_dst_mem}, {DNNL_ARG_TO, tensor_mgr_->getTensorBagAt(index_in_network_)->dst_mem_}});
            //         s_.wait();
            //         timer_->recordOpTimeTaken(index_in_network_, calculate_op_time_taken(start), "REORDER DEST");
            //     }
            // }

            Tracer::func_end("ConvLayer::doConv");    
        } 

        
        void ConvLayer::printStatus() {
            Tracer::func_begin("ConvLayer::printStatus");


            Tracer::func_end("ConvLayer::printStatus");          
        }

        ConvLayer::~ConvLayer() {
            Tracer::func_begin("ConvLayer::~ConvLayer");

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

            //dnnl_memory_destroy(tensor_mgr_->getTensorBagAt(index_in_network_)->src_mem_);
            //if(tensor_mgr_->getTensorBagAt(index_in_network_)->src_mem_.get_data_handle()) {
            //    free(tensor_mgr_->getTensorBagAt(index_in_network_)->src_mem_.get_data_handle());
            // }

            Tracer::func_end("ConvLayer::~ConvLayer");    
            
        }
    }
}
