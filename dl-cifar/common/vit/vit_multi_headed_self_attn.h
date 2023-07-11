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

#ifndef DL_CIFAR_MULTI_HEADED_ATTENTION_H_
#define DL_CIFAR_MULTI_HEADED_ATTENTION_H_

#include <vector>
#include "timing.h"
#include "mlp.h"
#include "vit_self_attn_head.h"
#include "vit_self_attn_head.h"
#include "tracing.h"
#include "handle.h"
namespace dl_cifar::common {
    class VitMultiHeadedSelfAttn {
        private:
            Timer* timer_;
            LangHandle *langHandle_;

            //std::vector<VitSelfAttnHeadWithoutBatch *> heads_;
            std::vector<VitSelfAttnHead *> heads_;
            std::vector<float *> d_intermediateZs_, d_d_intermediateZs_;
            int noOfHeads_;

            int batchSize_, embSize_, intermediateEmbSize_, noOfEmbs_;

            float *d_X_, *d_Z_;
            float *d_dX_, *d_dZ_;
            int concatIntermediateZSize_;
            float *d_concatIntermediateZ_, *d_d_concatIntermediateZ_;

            int oWeightsSize_;
            float *h_oWeights_, *d_oWeights_, *d_d_oWeights_;

        public:
            VitMultiHeadedSelfAttn(LangHandle *langHandle, Timer* timer, int batchSize, int embedSize, 
                            int intermediateEmbSize, int noOfEmbeds, int noOfHeads, float *d_X, float *d_Z, float *d_dX, float *d_dZ);

            void doFw();
            void doBw();

            ~VitMultiHeadedSelfAttn();

    };

    class VitMultiHeadedSelfAttnController {
        public:
            static void execute() {
                Timer* timer = new Timer();
                
                LangHandle *langHandle = new LangHandle(timer);

                int batchSize = 1024;
                int embedSize = 512;
                int noOfEmbeds = 4;
                int noOfHeads = 8;
                int intermediateEmbSize = embedSize/noOfHeads;    // 512/8 = 64;


                //---------- create input X -----------------------
                int patchEmbsSize = batchSize * noOfEmbeds * embedSize;

                float *h_X   = (float*)calloc(patchEmbsSize,   sizeof(float));  
                float *d_X, *d_dX;
                d_X = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));
                d_dX = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));

                langHandle->memCpyH2D(d_X, h_X, sizeof(float) * patchEmbsSize, true);

                //---------- create output Z -----------------------
                int Z_size = batchSize * noOfEmbeds * embedSize;

                float *h_dZ   = (float*)calloc(Z_size,   sizeof(float));  
                float *d_Z, *d_dZ;            
                d_Z = langHandle->allocDevMem((Z_size) * sizeof(float));
                d_dZ = langHandle->allocDevMem((Z_size) * sizeof(float));
                
                VitMultiHeadedSelfAttn *vitMultiHeadedAttn = new VitMultiHeadedSelfAttn(langHandle, timer, batchSize, embedSize, intermediateEmbSize, 
                                                                            noOfEmbeds, noOfHeads, d_X, d_Z, d_dX, d_dZ);

                for(int i=0; i<3; i++) {
                    ImageProcessor::initImage(h_X, patchEmbsSize);
                    langHandle->memCpyH2D(d_X, h_X, sizeof(float) * patchEmbsSize, true);
                    vitMultiHeadedAttn->doFw();

                    ImageProcessor::initImage(h_dZ, Z_size);
                    langHandle->memCpyH2D(d_dZ, h_dZ, sizeof(float) * Z_size, true);
                    vitMultiHeadedAttn->doBw();
                }
                
            }
    };
};
#endif