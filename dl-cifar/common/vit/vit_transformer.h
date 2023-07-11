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

#ifndef DL_CIFAR_TRANSFORMER_H_
#define DL_CIFAR_TRANSFORMER_H_

#include <vector>
#include "timing.h"
#include "mlp.h"
#include "vit_transformer_encoder.h"
#include "tracing.h"
#include "handle.h"
namespace dl_cifar::common {
    class VitTransformer {
        private:
            int noOfEncoders_;
            std::vector<VitTransformerEncoder *> encoders_;

        public:
            VitTransformer(LangHandle *langHandle, Timer* timer, int noOfEncoders, int batchSize, 
                int embSize, int embByNoOfHeadsSize, int noOfEmbs, int noOfHeads, float *d_patchEmbs, float *d_d_patchEmbs);

            void doFw();
            void doBw();

            ~VitTransformer();


    };

    class VitTransformerController {
        public:
            static void execute() {
                Timer* timer = new Timer();
                
                LangHandle *langHandle = new LangHandle(timer);

                int batchSize = 1024;
                int embSize = 512;
                int noOfEmbs = 4;
                int noOfHeads = 8;
                int embByNoOfHeadsSize = embSize/noOfHeads;    // 512/8 = 64;

                //---------- create patch embedding -----------------------
                int patchEmbsSize = batchSize * noOfEmbs * embSize;
                float *h_patchEmbs   = (float*)calloc(patchEmbsSize,   sizeof(float));  
                float *h_d_patchEmbs   = (float*)calloc(patchEmbsSize,   sizeof(float));  
                float *d_patchEmbs, *d_d_patchEmbs;
                d_patchEmbs = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));
                d_d_patchEmbs = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));


                int noOfEncoders = 3;
                VitTransformer *transformer = new VitTransformer(langHandle, timer, noOfEncoders, batchSize, embSize, embByNoOfHeadsSize, 
                                                            noOfEmbs, noOfHeads, d_patchEmbs, d_d_patchEmbs);

                int iterations = 3; 
                for(int i=0; i<iterations; i++) {
                    ImageProcessor::initImage(h_patchEmbs, patchEmbsSize);
                    langHandle->memCpyH2D(d_patchEmbs, h_patchEmbs, sizeof(float) * patchEmbsSize, true);
                    transformer->doFw();

                    ImageProcessor::initImage(h_d_patchEmbs, patchEmbsSize);
                    langHandle->memCpyH2D(d_d_patchEmbs, h_d_patchEmbs, sizeof(float) * patchEmbsSize, true);
                    transformer->doBw();

                }
            }

    };
};
#endif
