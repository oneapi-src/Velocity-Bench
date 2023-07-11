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

#ifndef DL_CIFAR_MULTI_HEADED_CLASS_ATTENTION_H_
#define DL_CIFAR_MULTI_HEADED_CLASS_ATTENTION_H_

#include <vector>
#include <cassert>
#include "timing.h"
#include "mlp.h"
#include "cait_class_attn_head.h"
#include "tracing.h"
#include "handle.h"

namespace dl_cifar::common {

    class CaitMultiHeadedClassAttn {
        private:
            Timer* timer_;
            LangHandle *langHandle_;

            std::vector<CaitClassAttnHead *> heads_;
            std::vector<float *> d_intermediateZs_, d_d_intermediateZs_;
            int noOfHeads_;

            int batchSize_, embSize_, intermediateEmbSize_;
            int noOfEmbs_;

            const float * const d_inputPatchEmbs_;
            float *d_inputClsEmb_, *d_outputAttnResiual_;
            int concatIntermediateZSize_;
            float *d_concatIntermediateZ_, *d_d_concatIntermediateZ_;

            float *d_d_inputPatchEmbs_, *d_d_inputClsEmb_, *d_d_outputAttnResiual_;

            int oWeightsSize_;
            float *h_oWeights_, *d_oWeights_;

        public:
            CaitMultiHeadedClassAttn(LangHandle *langHandle, Timer* timer, int batchSize, int embSize, int intermediateEmbSize, int noOfEmbs,
                            int noOfHeads, const float * const d_inputPatchEmbs, float *d_d_inputPatchEmbs, float *d_inputClsEmb, float *d_d_inputClsEmb,
                            float *d_outputAttnResiual, float *d_d_outputAttnResiual);

            void doFw();
            void doBw();

            ~CaitMultiHeadedClassAttn();


    };

    class MultiHeadedClassAttnController {
        public:
            static void execute() {
                Timer* timer = new Timer();
                
                LangHandle *langHandle = new LangHandle(timer);

                int batchSize = 1024;
                int embSize = 768;  //512;
                int noOfEmbs = 4;
                int noOfHeads = embSize/48;   //16    //  d=48xh  (from the paper)
                int intermediateEmbSize = embSize/noOfHeads;     //768/16 = 48            // 512/8 = 64;


                //---------- input patch embeddings -----------------------
                int patchEmbsSize = batchSize * noOfEmbs * embSize;

                float *h_inputPatchEmbs   = (float*)calloc(patchEmbsSize,   sizeof(float));  
                float *d_inputPatchEmbs, *d_d_inputPatchEmbs;
                d_inputPatchEmbs = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));
                d_d_inputPatchEmbs = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));


                //---------- create input class emb -----------------------
                int clsSize = batchSize * 1 * embSize;
                float *h_inputClsEmb   = (float*)calloc(clsSize,   sizeof(float));  
                float *d_inputClsEmb, *d_d_inputClsEmb;
                d_inputClsEmb = langHandle->allocDevMem((clsSize) * sizeof(float));
                d_d_inputClsEmb = langHandle->allocDevMem((clsSize) * sizeof(float));


                //---------- create residual emb -----------------------
                float *d_outputAttnResiual_, *d_d_outputAttnResiual_;
                int outputAttnResuidualSize = batchSize * 1 * embSize;
                float *h_d_outputAttnResiual_   = (float*)calloc(outputAttnResuidualSize,   sizeof(float));  
                d_outputAttnResiual_ = langHandle->allocDevMem((outputAttnResuidualSize) * sizeof(float));
                d_d_outputAttnResiual_ = langHandle->allocDevMem((outputAttnResuidualSize) * sizeof(float));

                
                CaitMultiHeadedClassAttn *multiHeadedClassAttn = new CaitMultiHeadedClassAttn(langHandle, timer, batchSize, embSize, 
                                                        intermediateEmbSize, noOfEmbs, noOfHeads, d_inputPatchEmbs, d_d_inputPatchEmbs, d_inputClsEmb,
                                                        d_d_inputClsEmb, d_outputAttnResiual_, d_d_outputAttnResiual_);

                int iter = 1;
                for(int i=0; iter<5; i++) {
                    ImageProcessor::initImage(h_inputPatchEmbs, patchEmbsSize);
                    ImageProcessor::initImage(h_inputClsEmb, clsSize);
                    langHandle->memCpyH2D(d_inputPatchEmbs, h_inputPatchEmbs, sizeof(float) * patchEmbsSize, false);
                    langHandle->memCpyH2D(d_inputClsEmb, h_inputClsEmb, sizeof(float) * clsSize, false);
                    langHandle->synchronize();
                    multiHeadedClassAttn->doFw();

                    ImageProcessor::initImage(h_d_outputAttnResiual_, outputAttnResuidualSize);
                    langHandle->memCpyH2D(d_d_outputAttnResiual_, h_d_outputAttnResiual_, sizeof(float) * outputAttnResuidualSize, true);
                    multiHeadedClassAttn->doBw();
                }
                
            }
    };
};
#endif