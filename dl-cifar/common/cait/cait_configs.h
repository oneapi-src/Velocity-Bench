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

#ifndef DL_CIFAR_CAIT_CONFIGS_H_
#define DL_CIFAR_CAIT_CONFIGS_H_

#include <string>

namespace dl_cifar::common {
    struct CaitParams {
        int batchSize;
        int noOfSAEncoders;
        int noOfCAEncoders;
        int imgWidth;
        int imgHeight;
        int imgNoOfChannels;
        int patchSize;
        int noOfHeads;
        int embSize;
    };

    class CaitConfigs {
        public:
            inline static const std::string cifar_dataset_dir = "../../datasets/cifar-10-batches-bin/";
            inline static const std::string cifar_dataset_file = cifar_dataset_dir + "data_batch_1.bin";

            static constexpr int cifarNoOfChannels = 3;
            static constexpr int cifarImgWidth     = 32;
            static constexpr int cifarImgHeight    = 32;

            static constexpr CaitParams caitM36_224_lowmemGPU = {
                1,         // batchSize
                1,         // noOfSAEncoders
                1,         // noOfCAEncoders
                224,       // imgWidth
                224,       // imgHeight
                3,         // imgNoOfChannels
                16,        // patchSize
                16,        // noOfHeads
                768        // embSize
            };

            static constexpr CaitParams caitM36_224_workload_default = {
                4,        // batchSize
                36,         // noOfSAEncoders
                2,         // noOfCAEncoders
                224,       // imgWidth
                224,       // imgHeight
                3,         // imgNoOfChannels
                16,        // patchSize
                16,        // noOfHeads
                768        // embSize
            };

            static constexpr CaitParams caitM36_224_fullsize = {
                1024,      // batchSize
                36,        // noOfSAEncoders
                2,         // noOfCAEncoders
                224,       // imgWidth
                224,       // imgHeight
                3,         // imgNoOfChannels
                16,        // patchSize
                16,        // noOfHeads
                768        // embSize
            };
    };
};

#endif