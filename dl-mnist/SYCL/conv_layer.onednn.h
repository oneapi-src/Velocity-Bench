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

#ifndef DEEP_LEARNING_DPCPP_CONV_LAYER_H_
#define DEEP_LEARNING_DPCPP_CONV_LAYER_H_

#include "oneapi/dnnl/dnnl.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "dnnl_sycl.hpp"
#endif
#include "../common/timing.h"
#include "../common/workload_params.h"
#include "tensor_mgr.onednn.h"


using namespace dnnl;
using namespace dl_infra::common;

namespace dl_infra {
    namespace onednn {
        class IConvLayer {
            public:
             virtual void doConv(double prev_cum_time_taken) = 0; 
        };

        class ConvLayer: public IConvLayer {
            private:
                int index_in_network_, total_layers_in_nw_;
                Timer* timer_;
                engine *eng_;
                stream *s_;
                
                TensorMgr* tensor_mgr_;

                memory conv_scratchpad_mem_;
                convolution_forward conv_;
                convolution_forward::primitive_desc conv_pd;
               
                int* input_tensor_dims_;
                int* filter_tensor_dims_;
                int* output_tensor_dims_;

                int dilation_[2] {1, 1};
                int pad_[2] {0, 0};
                int conv_stride_[2] {1, 1};

                bool initialized_ = false;

                IConvLayer* nextConvLayer_ = NULL;
                double totalTime = 0;
                bool add_mem_transfer_time_ = false;

                bool need_reorder_src_ = false;
                bool need_reorder_weights_ = false;
                bool need_reorder_dst_ = false;

                void write_to_dnnl_memory(void *handle, dnnl::memory &mem);

                friend class DlNetworkMgr;
                WorkloadParams* workloadParams_;

                bool dryRun_ = false;
                
            public:
                ConvLayer(WorkloadParams* workloadParams, int index_in_network, int total_layers_in_nw, 
                    Timer* timer, TensorMgr* tensor_mgr, engine *eng, stream *s, 
                    int input_tensor_dims[], int filter_tensor_dims[], int output_tensor_dims[]);
                ConvLayer(WorkloadParams* workloadParams, int index_in_network, int total_layers_in_nw, 
                    Timer* timer, TensorMgr* tensor_mgr, IConvLayer* nextConvLayer, engine *eng, stream *s, 
                    int input_tensor_dims[], int filter_tensor_dims[], int output_tensor_dims[]);
                ~ConvLayer();

                void initialize();
                void doIOTensorAndWSAllocs();
                void doTensorAndWSDeallocs();
                void stage1Cleanup();
                void findBestAlgo();
                
                void doConv(double prev_cum_time_taken); 
                void printStatus();
                double getTotalTime() {return totalTime;};
                void setDryRun(bool dryRun) { dryRun_ = dryRun; }

            private:
         
                void createWorkspace();
                void createTensorDescriptors();
                void createTensors();
                void reorderWeightsIfRequired();

                void calculateStrideDims();                
        };
    }
}


#endif