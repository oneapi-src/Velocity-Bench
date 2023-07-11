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

#include "dl_network_mgr.onednn.h"

#include "../common/timing.h"
#include "../common/workload_params.h"
#include <vector>
#include <map>

using namespace dl_infra::common;
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

namespace dl_infra {
    namespace onednn {
        std::vector<ConvLayer*> DlNetworkMgr::createAllLayers(string networkName, int no_of_conv_layers, int *conv_dims, TensorMgr* tensoMgr) {
            std::vector<ConvLayer*> convLayers(no_of_conv_layers);
            auto conv_dims_arr = (int(*)[no_of_params_][no_of_tensor_dims_]) (conv_dims);

            for(int i=0; i<no_of_conv_layers; i++) {
                convLayers[i] = new ConvLayer(workloadParams_, i, no_of_conv_layers, timer_, tensoMgr, eng_, s_,  
                                conv_dims_arr[i][0], conv_dims_arr[i][1], conv_dims_arr[i][2]); 
            }
            return convLayers;
        }

        void DlNetworkMgr::initializeNetwork(string networkName) {
            DlNetwork* dlNetwork = networkMap[networkName];
            
            // ---do network-wide init---
            // 1. call stage1Init() on each layer
            // 2. do parent level config/init

            for(int i=0; i<dlNetwork->convLayers_.size(); i++) {
                dlNetwork->convLayers_[i]->initialize();
            }
            // for(int i=0; i<dlNetwork->convLayers_.size(); i++) {
            //     if(dlNetwork->convLayers_[i]->conv_pd.weights_desc() !=  tensor_mgr_->getTensorBagAt(index_in_network_)->weights_mem_.get_desc()) {
            //         need_reorder_weights_ = true;
            //     }
            // }
            for(int i=1; i<dlNetwork->convLayers_.size(); i++) {
                if(dlNetwork->convLayers_[i-1]->conv_pd.dst_desc() !=  dlNetwork->convLayers_[i]->conv_pd.src_desc()) {
                    std::cout << "\n\n\ninitializeNetwork " << i << " need_reorder_src_: 1\n\n\n" << std::endl;
                    //dlNetwork->convLayers_[i]->need_reorder_src_ = true;
                }
            }          

            if(workloadParams_->getDryRun()==true) {
                for(int i=0; i<dlNetwork->convLayers_.size(); i++) {
                    dlNetwork->convLayers_[i]->setDryRun(true);
                }
                if(tensorMgr==0) {
                    throw runtime_error("Attempt to use TensorMgr before creation"); 
                } else {
                    tensorMgr->setDryRun(true);
                }

                executeInferenceRun(networkName);

                for(int i=0; i<dlNetwork->convLayers_.size(); i++) {
                    dlNetwork->convLayers_[i]->setDryRun(false);
                }
                if(tensorMgr==0) {
                    throw runtime_error("Attempt to use TensorMgr before creation"); 
                } else {
                    tensorMgr->setDryRun(false);
                }
            }         

        }

        void DlNetworkMgr::createDLNetwork(string networkName, int no_of_conv_layers, int *conv_dims) {
            tensorMgr = new TensorMgr(workloadParams_, timer_, dataFileReadTimer_, no_of_conv_layers, eng_);
            std::vector<ConvLayer*> convLayers = createAllLayers(networkName, no_of_conv_layers, conv_dims, tensorMgr);
            DlNetwork* dlNetwork = new DlNetwork(networkName, no_of_conv_layers, conv_dims, std::move(convLayers), tensorMgr);

            networkMap.insert({networkName, dlNetwork});
            initializeNetwork(networkName);
        }

        void DlNetworkMgr::executeInferenceRun(string networkName) {
            DlNetwork* dlNetwork = networkMap[networkName];

            //std::cout << "Starting new run" << std::endl;
            if(workloadParams_->getTensorMemPolicy()==WorkloadParams::TensorMemPolicy::ALL_MEM_ALLOC_AT_START) {
                for(int i=0; i<dlNetwork->noOfConvLayers_; i++) {
                    dlNetwork->convLayers_[i]->doIOTensorAndWSAllocs();
                }
                for(int i=0; i<dlNetwork->noOfConvLayers_; i++) {
                    dlNetwork->convLayers_[i]->doConv(0.0);
                }
                for(int i=0; i<dlNetwork->noOfConvLayers_; i++) {
                    dlNetwork->convLayers_[i]->doTensorAndWSDeallocs();
                } 
            } else if(workloadParams_->getTensorMemPolicy()==WorkloadParams::TensorMemPolicy::MEM_ALLOC_DEALLOC_EVERY_CONV) {
                for(int i=0; i<dlNetwork->noOfConvLayers_; i++) {
                    dlNetwork->convLayers_[i]->doIOTensorAndWSAllocs();
                    dlNetwork->convLayers_[i]->doConv(0.0);
                    dlNetwork->convLayers_[i]->doTensorAndWSDeallocs();
                }
            } else {
                throw std::runtime_error("TensorMemPolicy is not ALL_MEM_ALLOC_AT_START or MEM_ALLOC_DEALLOC_EVERY_CONV"); 
            }
        }

    }
}