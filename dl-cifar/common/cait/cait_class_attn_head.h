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

#ifndef DL_CIFAR_CLASS_ATTENTION_HEAD_H_
#define DL_CIFAR_CLASS_ATTENTION_HEAD_H_

#include <vector>
#include <cmath>
#include "timing.h"
#include "mlp.h"
#include "tracing.h"
#include "handle.h"
#include "basic-dl/softmax_layer.h"

namespace dl_cifar::common {
    class CaitClassAttnHead {
        private:
            Timer* timer_;
            LangHandle *langHandle_;

            size_t batchSize_;
            size_t embSize_;
            size_t intermediateEmbSize_;
            size_t noOfEmbs_;

            size_t qkvWeightsSize_, clsEmbPlusPatchEmbSize_;
            size_t qSize_, kAndVSize_, QKt_size_;

            float *h_qWeights_, *h_kWeights_, *h_vWeights_; 
            float *d_qWeights_, *d_kWeights_, *d_vWeights_; 
            float *d_Q_, *d_K_, *d_V_; 
            float *d_QKt_, *d_d_QKt_;
            const float * const d_inputPatchEmbs_;
            float *d_inputClsEmb_, *d_outputClsEmbPerHead_, *d_clsEmbPlusPatchEmbs_, *d_d_clsEmbPlusPatchEmbs_;
            float *d_d_qWeights_, *d_d_kWeights_, *d_d_vWeights_, *d_dQ_, *d_dK_, *d_dV_;
            float scale_, scale_bw_;

            float *d_d_inputPatchEmbs_, *d_d_inputClsEmb_, *d_d_inputClsEmb_Vpath_, *d_d_outputClsEmbPerHead_;

            SoftmaxLayer *softmaxLayer_;
            

        public:
            CaitClassAttnHead(LangHandle *langHandle, Timer* timer, size_t batchSize, size_t embSize, 
                                    size_t intermediateEmbSize, size_t noOfEmbs, const float * const d_inputPatchEmbs, float *d_d_inputPatchEmbs,
                                    float *d_inputClsEmb, float *d_d_inputClsEmb, float *d_outputClsEmbPerHead, float *d_d_outputClsEmbPerHead);    

            ~CaitClassAttnHead();

            void doFw();
            void doBw();

    };


    class ClassAttnHeadController {
        public:
            static void execute() {
                Timer* timer = new Timer();

                LangHandle *langHandle = new LangHandle(timer);

                size_t batchSize = 1024;
                size_t embSize = 768;  //512;
                size_t intermediateEmbSize = 48;   // IMPORTANT: should be d/h = 768/16 = 48, assuming h=16  
                size_t noOfEmbs = 4;

                //---------- create input patch embeddings -----------------------
                size_t patchEmbsSize = batchSize * noOfEmbs * embSize;

                float *h_inputPatchEmbs   = (float*)calloc(patchEmbsSize,   sizeof(float));  
                float *d_inputPatchEmbs, *d_d_inputPatchEmbs;
                d_inputPatchEmbs = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));
                d_d_inputPatchEmbs = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));


                //---------- create input class emb -----------------------
                size_t clsSize = batchSize * 1 * embSize;
                float *h_inputClsEmb   = (float*)calloc(clsSize,   sizeof(float));  
                float *d_inputClsEmb, *d_d_inputClsEmb;
                d_inputClsEmb = langHandle->allocDevMem((clsSize) * sizeof(float));
                d_d_inputClsEmb = langHandle->allocDevMem((clsSize) * sizeof(float));


                //---------- create output class emb -----------------------
                float *d_outputClsEmb, *d_d_outputClsEmb;
                size_t outputClsSize = batchSize * 1 * intermediateEmbSize;
                float *h_d_outputClsEmb   = (float*)calloc(outputClsSize,   sizeof(float));  
                d_outputClsEmb = langHandle->allocDevMem((outputClsSize) * sizeof(float));
                d_d_outputClsEmb = langHandle->allocDevMem((outputClsSize) * sizeof(float));


                CaitClassAttnHead *classAttnHead = new CaitClassAttnHead(langHandle, timer, batchSize, embSize, 
                                                                                intermediateEmbSize, noOfEmbs, d_inputPatchEmbs, d_d_inputPatchEmbs,
                                                                                d_inputClsEmb, d_d_inputClsEmb, d_outputClsEmb, d_d_outputClsEmb);

                for(int i=0; i<3; i++) {
                    ImageProcessor::initImage(h_inputPatchEmbs, patchEmbsSize);
                    langHandle->memCpyH2D(d_inputPatchEmbs, h_inputPatchEmbs, sizeof(float) * patchEmbsSize, false);
                    ImageProcessor::initImage(h_inputClsEmb, clsSize);
                    langHandle->memCpyH2D(d_inputClsEmb, h_inputClsEmb, sizeof(float) * clsSize, false);
                    langHandle->synchronize();

                    classAttnHead->doFw();



                    ImageProcessor::initImage(h_d_outputClsEmb, outputClsSize);
                    langHandle->memCpyH2D(d_d_outputClsEmb, h_d_outputClsEmb, sizeof(float) * outputClsSize, true);

                    classAttnHead->doBw();
                }

            }
    };
};
#endif