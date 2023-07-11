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

#ifndef DL_CIFAR_SELF_ATTENTION_HEAD_H_
#define DL_CIFAR_SELF_ATTENTION_HEAD_H_

#include <vector>
#include <cmath>
#include "timing.h"
#include "mlp.h"
#include "tracing.h"
#include "handle.h"
#include "basic-dl/softmax_layer.h"


namespace dl_cifar::common {
    class VitSelfAttnHead {
        private:
            Timer* timer_;
            LangHandle *langHandle_;

            int batchSize_;
            int embSize_;
            int intermediateEmbSize_;
            int noOfEmbs_;

            int qkvWeightsSize_;
            int qkvSize_, QKt_size_;

            float *h_qWeights_, *h_kWeights_, *h_vWeights_; 
            float *d_qWeights_, *d_kWeights_, *d_vWeights_; 
            float *d_Q_, *d_K_, *d_V_; 
            float *d_QKt_, *d_d_QKt_;
            float *d_X_, *d_intermediateZ_;
            float *d_dX_, *d_dX_Kpath_, *d_dX_Vpath_, *d_d_intermediateZ_;
            float *d_d_qWeights_, *d_d_kWeights_, *d_d_vWeights_, *d_dQ_, *d_dK_, *d_dV_; 
            float scale_, scale_bw_;
        
            SoftmaxLayer *softmaxLayer_;
            

        public:
            VitSelfAttnHead(LangHandle *langHandle, Timer* timer, int batchSize, int embSize, int intermediateEmbSize,
                                    int noOfEmbs, float *d_X, float *d_intermediateZ, float *d_dX, float *d_d_intermediateZ);   

            ~VitSelfAttnHead();

            void doFw();
            void doBw();

    };


    class VitSelfAttnHeadController {
        public:
            static void execute() {
                Timer* timer = new Timer();
                            
                LangHandle *langHandle = new LangHandle(timer);

                int batchSize = 1024;
                int embSize = 512;
                int intermediateEmbSize = 64;
                int noOfEmbs = 4;

                //---------- create input X -----------------------
                int patchEmbsSize = batchSize * noOfEmbs * embSize;

                float *h_X   = (float*)calloc(patchEmbsSize,   sizeof(float));  
                ImageProcessor::initImage(h_X, patchEmbsSize);
                float *d_X, *d_dX;
                d_X = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));
                d_dX = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));
                langHandle->memCpyH2D(d_X, h_X, sizeof(float) * patchEmbsSize, true);

                //---------- create output Z -----------------------
                int intermediateZSize = batchSize * noOfEmbs * intermediateEmbSize;

                float *d_intermediateZ, *d_d_intermediateZ;
                float *h_d_intermediateZ   = (float*)calloc(intermediateZSize,   sizeof(float));  
                d_intermediateZ = langHandle->allocDevMem((intermediateZSize) * sizeof(float));
                d_d_intermediateZ = langHandle->allocDevMem((intermediateZSize) * sizeof(float));


                VitSelfAttnHead *vitSelfAttnHead = new VitSelfAttnHead(langHandle, timer, batchSize, embSize, 
                                                                intermediateEmbSize, noOfEmbs, d_X, d_intermediateZ, d_dX, d_d_intermediateZ);

                for(int i=0; i<5; i++) {
                    ImageProcessor::initImage(h_X, patchEmbsSize);
                    langHandle->memCpyH2D(d_X, h_X, sizeof(float) * patchEmbsSize, true);
                    vitSelfAttnHead->doFw();

                    ImageProcessor::initImage(h_d_intermediateZ, intermediateZSize);
                    langHandle->memCpyH2D(d_d_intermediateZ, h_d_intermediateZ, sizeof(float) * intermediateZSize, true);
                    vitSelfAttnHead->doBw();
                }

            }
    };
};
#endif
