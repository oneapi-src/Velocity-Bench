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

#ifndef DL_MNIST_LAYER_MGR_CUDA_H
#define DL_MNIST_LAYER_MGR_CUDA_H

#include "conv_layer.onednn.h"
#include "../common/timing.h"
#include "../common/workload_params.h"
#include <vector>
#include <map>

using namespace dl_infra::common;


namespace dl_infra {
    namespace onednn {
        class DlNetwork {
            private:
                string networkName_;
                int *convDims_;
                int noOfConvLayers_;
                std::vector<ConvLayer*> convLayers_;
                TensorMgr* tensorMgr_;

                friend class DlNetworkMgr;
            private:
                DlNetwork(string networkName, int noOfConvLayers, int *convDims, std::vector<ConvLayer*> convLayers, TensorMgr* tensorMgr): networkName_(std::move(networkName)), convDims_(convDims), noOfConvLayers_(noOfConvLayers), convLayers_(std::move(convLayers)), tensorMgr_(tensorMgr) {
                    
                }

        };
        class DlNetworkMgr {
            private:
                Timer* timer_, *dataFileReadTimer_;
                engine eng_;
                stream s_;
                WorkloadParams* workloadParams_;
                //WorkloadParams::TensorMemPolicy tensorMemPolicy_;

                const int no_of_params_ = 3;  //inputs, weights and output
                const int no_of_tensor_dims_ = 4;    //NCHW

                TensorMgr* tensorMgr;

                std::map<std::string, DlNetwork*> networkMap;

                std::vector<ConvLayer*> createAllLayers(string networkName, int no_of_conv_layers, int *conv_dims, TensorMgr* tensoMgr);
                void initializeNetwork(string networkName);

            public:
                DlNetworkMgr(WorkloadParams* workloadParams, engine eng, stream s, Timer* timer, Timer* dataFileReadTimer)
                    : workloadParams_(workloadParams), eng_(std::move(eng)), s_(std::move(s)), timer_(timer), tensorMgr(0), dataFileReadTimer_(dataFileReadTimer) {}
                void createDLNetwork(string networkName, int no_of_conv_layers, int *conv_dims);
                void executeInferenceRun(string networkName);
        };
    }
}


#endif
