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

#ifndef DL_CIFAR_TRANSFORMER_ENCODER_H_
#define DL_CIFAR_TRANSFORMER_ENCODER_H_

#include <vector>
#include <cmath>
#include "timing.h"
#include "mlp.h"
#include "vit_multi_headed_self_attn.h"
#include "basic-dl/lnorm_layer.h"
#include "tracing.h"
#include "handle.h"
namespace dl_cifar::common {
    class VitTransformerEncoder {
        private:
            Timer* timer_;
            LangHandle *langHandle_;

            int noOfEmbs_;
            int batchSize_, embSize_, intermediateEmbSize_;

            float *d_patchEmbs_, *d_d_patchEmbs_;
            float *d_lNormedPatchEmbs1_, *d_d_lNormedPatchEmbs1_, *d_d_norm1Output_;
            float *d_attnResidual_, *d_d_attnResidual_;
            float *d_lNormedPatchEmbs2_, *d_d_lNormedPatchEmbs2_, *d_d_norm2Output_;
            float *d_mlpResidual_, *d_d_mlpResidual_;

            VitMultiHeadedSelfAttn *multiHeadedAttention_;
            LNormLayer *lNormLayer1_, *lNormLayer2_;
            Mlp *mlp_;


        public:
            VitTransformerEncoder(LangHandle *langHandle, Timer* timer, int batchSize, int embSize, 
                    int intermediateEmbSize, int noOfEmbs, int noOfHeads, float *d_patchEmbs, float *d_d_patchEmbs);

            void doFw();
            void doBw();

            ~VitTransformerEncoder();



    };

    class VitTransformerEncoderController {
        public:
            static void execute() {
                Timer* timer = new Timer();
                
                LangHandle *langHandle = new LangHandle(timer);

                int batchSize = 1024;
                int embSize = 512;
                int noOfEmbs = 4;
                int noOfHeads = 8;
                int intermediateEmbSize = embSize/noOfHeads;    // 512/8 = 64;


                //---------- create patch embedding -----------------------
                int patchEmbsSize = batchSize * noOfEmbs * embSize;
                float *h_patchEmbs   = (float*)calloc(patchEmbsSize, sizeof(float));  
                float *h_d_patchEmbs = (float*)calloc(patchEmbsSize, sizeof(float));  
                float *d_patchEmbs, *d_d_patchEmbs;
                d_patchEmbs = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));
                d_d_patchEmbs = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));




                VitTransformerEncoder *transformerEncoder = new VitTransformerEncoder(langHandle, timer, batchSize, embSize, intermediateEmbSize,
                                                                                    noOfEmbs, noOfHeads, d_patchEmbs, d_d_patchEmbs);




                int iterations = 3; 
                for(int i=0; i<iterations; i++) {
                    ImageProcessor::initImage(h_patchEmbs, patchEmbsSize);
                    langHandle->memCpyH2D(d_patchEmbs, h_patchEmbs, sizeof(float) * patchEmbsSize, true);
                    transformerEncoder->doFw();

                    ImageProcessor::initImage(h_d_patchEmbs, patchEmbsSize);
                    langHandle->memCpyH2D(d_d_patchEmbs, h_d_patchEmbs, sizeof(float) * patchEmbsSize, true);
                    transformerEncoder->doBw();
                }

            }
    };
}


#endif