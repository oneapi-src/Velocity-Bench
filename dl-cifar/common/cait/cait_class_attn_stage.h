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

#ifndef DL_CIFAR_CAIT_CLASS_ATTENTION_STAGE_H_
#define DL_CIFAR_CAIT_CLASS_ATTENTION_STAGE_H_

#include <vector>
#include <cmath>
#include "timing.h"
#include "mlp.h"                                                                                                                                                                                     
#include "cait_multi_headed_class_attn.h"
#include "cait_class_attn_stage_encoder.h"
#include "tracing.h"
#include "handle.h"

namespace dl_cifar::common {

    class CaitClassAttnStage {
        private:
        Timer* timer_;
            LangHandle *langHandle_;

            std::vector<CaitClassAttnStageEncoder *> encoders_;
            int noOfEncoders_;


        public:
            CaitClassAttnStage(LangHandle *langHandle, Timer* timer, int noOfEncoders, int batchSize, 
                int embSize, int embByNoOfHeadsSize, int noOfEmbs, int noOfHeads, const float * const d_patchEmbs, float *d_d_patchEmbs,
                float *d_clsEmb, float *d_d_clsEmb)
                : langHandle_(langHandle), timer_(timer), noOfEncoders_(noOfEncoders) {

                Tracer::func_begin("CaitClassAttnStage::CaitClassAttnStage");

                for(int i=0; i<noOfEncoders_; i++) {
                    encoders_.push_back(new CaitClassAttnStageEncoder(langHandle, timer, batchSize, embSize, embByNoOfHeadsSize, 
                                                                        noOfEmbs, noOfHeads, d_patchEmbs, d_d_patchEmbs, d_clsEmb, d_d_clsEmb));     
                }

                Tracer::func_end("CaitClassAttnStage::CaitClassAttnStage");    
            }

            void doFw() {
                Tracer::func_begin("CaitClassAttnStage::doFw");

                for(int i=0; i<noOfEncoders_; i++) {
                    encoders_[i]->doFw();
                }

                Tracer::func_end("CaitClassAttnStage::doFw");        
            }

            void doBw() {
                Tracer::func_begin("CaitClassAttnStage::doBw");

                for(int i=0; i<noOfEncoders_; i++) {
                    encoders_[i]->doBw();
                }

                Tracer::func_end("CaitClassAttnStage::doBw");    
            }

            ~CaitClassAttnStage() {
                Tracer::func_begin("CaitClassAttnStage::~CaitClassAttnStage");
                for(int i=0; i<noOfEncoders_; i++) {
                    delete encoders_[i];
                }
                Tracer::func_end("CaitClassAttnStage::~CaitClassAttnStage");    
            }



    };


    class ClassAttnStageController {
        public:
            static void execute() {
                Timer* timer = new Timer();

                LangHandle *langHandle = new LangHandle(timer);

                int batchSize = 1024;
                int embSize = 768;  //512;
                int noOfEmbs = 4;
                int noOfHeads = 16;
                int embByNoOfHeadsSize = embSize/noOfHeads;   // IMPORTANT: should be d/h = 768/16 = 48, assuming h=16  

                

                //---------- create patch embedding -----------------------
                int patchEmbsSize = batchSize * noOfEmbs * embSize;
                float *h_d_patchEmbs = (float*)calloc(patchEmbsSize,   sizeof(float));  
                float *h_patchEmbs   = (float*)calloc(patchEmbsSize,   sizeof(float));  
                float *d_patchEmbs, *d_d_patchEmbs;
                d_patchEmbs = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));
                d_d_patchEmbs = langHandle->allocDevMem((patchEmbsSize) * sizeof(float));

                //---------- create class embedding -----------------------
                int clsEmbSize = batchSize * 1 * embSize;
                float *h_clsEmb   = (float*)calloc(clsEmbSize,   sizeof(float));  
                float *h_d_clsEmb = (float*)calloc(clsEmbSize,   sizeof(float));  
                float *d_clsEmb, *d_d_clsEmb;
                d_clsEmb = langHandle->allocDevMem((clsEmbSize) * sizeof(float));
                d_d_clsEmb = langHandle->allocDevMem((clsEmbSize) * sizeof(float));



                int noOfEncoders = 2;
                
                CaitClassAttnStage *classAttnStage = new CaitClassAttnStage(langHandle, timer, noOfEncoders, batchSize,
                                                        embSize, embByNoOfHeadsSize, noOfEmbs, noOfHeads, d_patchEmbs, d_d_patchEmbs,
                                                        d_clsEmb, d_d_clsEmb);


                int iterations = 2; 
                for(int i=0; i<iterations; i++) {
                    ImageProcessor::initImage(h_patchEmbs, patchEmbsSize);
                    ImageProcessor::initImage(h_clsEmb, clsEmbSize);
                    langHandle->memCpyH2D(d_patchEmbs, h_patchEmbs, sizeof(float) * patchEmbsSize, false);
                    langHandle->memCpyH2D(d_clsEmb, h_clsEmb, sizeof(float) * clsEmbSize, false);
                    langHandle->synchronize();
                    classAttnStage->doFw();

                    ImageProcessor::initImage(h_d_patchEmbs, patchEmbsSize);
                    ImageProcessor::initImage(h_d_clsEmb, clsEmbSize);
                    langHandle->memCpyH2D(d_d_patchEmbs, h_d_patchEmbs, sizeof(float) * patchEmbsSize, false);
                    langHandle->memCpyH2D(d_d_clsEmb, h_d_clsEmb, sizeof(float) * clsEmbSize, false);
                    langHandle->synchronize();
                    classAttnStage->doBw();




                }
            }

    };
};
#endif