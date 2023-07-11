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

#ifndef DL_CIFAR_VIT_CONFIGS_H_
#define DL_CIFAR_VIT_CONFIGS_H_

#include <string>

namespace dl_cifar::common {
    struct VitParams {
        int batchSize;
        int noOfEncoders;
        int imgWidth;
        int imgHeight;
        int imgNoOfChannels;
        int patchSize;
        int noOfHeads;
        int embSize;
        int mLPSize;
    };

    class VitConfigs {
        public:
            inline static const std::string cifar_dataset_dir = "../../datasets/cifar-10-batches-bin/";
            inline static const std::string cifar_dataset_file = cifar_dataset_dir + "data_batch_1.bin";

            static constexpr int cifarNoOfChannels = 3;
            static constexpr int cifarImgWidth     = 32;
            static constexpr int cifarImgHeight    = 32;

            static constexpr VitParams vitL16_params_lowmemGPU = {
                1,         // batchSize
                1,         // noOfLayers
                384,       // imgWidth
                384,       // imgHeight
                3,         // imgNoOfChannels
                16,        // patchSize
                16,        // noOfHeads
                1024,      // embSize
                4096       // mLPSize
            };

            static constexpr VitParams vitL16_params_workload_default = {
                4,       // batchSize
                24,         // noOfLayers
                384,       // imgWidth
                384,       // imgHeight
                3,         // imgNoOfChannels
                16,        // patchSize
                16,        // noOfHeads
                1024,      // embSize
                4096       // mLPSize
            };
        
            static constexpr VitParams vitL16_params_fullsize = {
                512,       // batchSize
                24,        // noOfLayers
                384,       // imgWidth
                384,       // imgHeight
                3,         // imgNoOfChannels
                16,        // patchSize
                16,        // noOfHeads
                1024,      // embSize
                4096       // mLPSize
            };



            static constexpr VitParams vitH14_params_lowmemGPU = {
                    1,         // batchSize
                    1,         // noOfLayers
                    384,       // imgWidth
                    384,       // imgHeight
                    3,         // imgNoOfChannels
                    14,        // patchSize
                    16,        // noOfHeads
                    1280,      // embSize
                    5120       // mLPSize
                };

            static constexpr VitParams vitH14_params_workload_default = {
                    4,       // batchSize
                    32,         // noOfLayers
                    384,       // imgWidth
                    384,       // imgHeight
                    3,         // imgNoOfChannels
                    14,        // patchSize
                    16,        // noOfHeads
                    1280,      // embSize
                    5120       // mLPSize
                };

            static constexpr VitParams vitH14_params_fullsize = {
                    512,       // batchSize
                    32,        // noOfLayers
                    384,       // imgWidth
                    384,       // imgHeight
                    3,         // imgNoOfChannels
                    14,        // patchSize
                    16,        // noOfHeads
                    1280,      // embSize
                    5120       // mLPSize
            };

    };
};

#endif