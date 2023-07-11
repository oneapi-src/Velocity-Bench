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

#ifndef DL_CIFAR_CLASS_ATTENTION_STAGE_ENCODER_H_
#define DL_CIFAR_CLASS_ATTENTION_STAGE_ENCODER_H_

#include <vector>
#include <cmath>
#include "blas_routines.h"
#include "mlp.h"
#include "cait_multi_headed_class_attn.h"
#include "tracing.h"
#include "handle.h"

namespace dl_cifar::common {

    class CaitClassAttnStageEncoder {
        private:
            LangHandle *langHandle_;

            int noOfEmbs_;
            int batchSize_, embSize_, intermediateEmbSize_;

            float *d_clsEmb_, *d_d_clsEmb_; 
            const float * const d_patchEmbs_;
            float *d_d_patchEmbs_;
            float *d_lNormedPatchEmbs_, *d_d_lNormedPatchEmbs_, *d_d_normedPatchOutput_;
            float *d_lNormedClsEmb1_,  *d_d_lNormedClsEmb1_,  *d_d_normedCls1Output_;
            float *d_lNormedClsEmb2_,  *d_d_lNormedClsEmb2_,  *d_d_normedCls2Output_;
            float *d_attnResidual_, *d_d_attnResidual_;
            float *d_mlpResidual_,  *d_d_mlpResidual_;

            CaitMultiHeadedClassAttn *caitMultiHeadedClassAttn_;
            LNormLayer *lNormPatchLayer_, *lNormClsLayer1_, *lNormClsLayer2_;
            Mlp *mlp_;

        public:
            CaitClassAttnStageEncoder(LangHandle *langHandle, Timer* timer, int batchSize,
                int embSize, int embByNoOfHeadsSize, int noOfEmbs, int noOfHeads, const float * const d_patchEmbs, float *d_d_patchEmbs,
                float *d_clsEmb, float *d_d_clsEmb)
                : langHandle_(langHandle), batchSize_(batchSize), embSize_(embSize), 
                intermediateEmbSize_(embByNoOfHeadsSize), noOfEmbs_(noOfEmbs), d_patchEmbs_(d_patchEmbs), d_d_patchEmbs_(d_d_patchEmbs),
                d_clsEmb_(d_clsEmb), d_d_clsEmb_(d_d_clsEmb)  {

                Tracer::func_begin("CaitClassAttnStageEncoder::CaitClassAttnStageEncoder");






                // -------------initialization for multi headed class attention --------------------------------------------
                int lNormedPatchEmbsSize = batchSize * noOfEmbs * embSize;
                d_lNormedPatchEmbs_ = langHandle->allocDevMem((lNormedPatchEmbsSize) * sizeof(float));
                d_d_lNormedPatchEmbs_ = langHandle->allocDevMem((lNormedPatchEmbsSize) * sizeof(float));
                d_d_normedPatchOutput_ = langHandle->allocDevMem((lNormedPatchEmbsSize) * sizeof(float));

                int lNormedClsEmb1Size = batchSize * 1 * embSize_;
                d_lNormedClsEmb1_ = langHandle->allocDevMem((lNormedClsEmb1Size) * sizeof(float));
                d_d_lNormedClsEmb1_ = langHandle->allocDevMem((lNormedClsEmb1Size) * sizeof(float));
                d_d_normedCls1Output_ = langHandle->allocDevMem((lNormedClsEmb1Size) * sizeof(float));


                lNormPatchLayer_ = new LNormLayer(langHandle, timer, d_patchEmbs_, d_lNormedPatchEmbs_, 
                                    d_d_normedPatchOutput_, d_d_lNormedPatchEmbs_, batchSize_, noOfEmbs_, embSize_);

                lNormClsLayer1_ = new LNormLayer(langHandle, timer, d_clsEmb_, d_lNormedClsEmb1_, 
                                    d_d_normedCls1Output_, d_d_lNormedClsEmb1_,1, 1, embSize_);


                int attnResiualSize = batchSize * 1 * embSize_;
                d_attnResidual_ = langHandle->allocDevMem((attnResiualSize) * sizeof(float));
                d_d_attnResidual_ = langHandle->allocDevMem((attnResiualSize) * sizeof(float));
                

                caitMultiHeadedClassAttn_ = new CaitMultiHeadedClassAttn(langHandle, timer, batchSize_, embSize_, intermediateEmbSize_, 
                                                            noOfEmbs_, noOfHeads, d_lNormedPatchEmbs_, d_d_lNormedPatchEmbs_, 
                                                            d_lNormedClsEmb1_, d_d_lNormedClsEmb1_, d_attnResidual_, d_d_normedCls1Output_);





                // -------------initialization for FFN  --------------------------------------------

                int lNormedClsEmb2Size = 1 * embSize_;
                d_lNormedClsEmb2_ = langHandle->allocDevMem((lNormedClsEmb2Size) * sizeof(float));
                d_d_lNormedClsEmb2_ = langHandle->allocDevMem((lNormedClsEmb2Size) * sizeof(float));
                d_d_normedCls2Output_ = langHandle->allocDevMem((lNormedClsEmb2Size) * sizeof(float));

                lNormClsLayer2_ = new LNormLayer(langHandle, timer, d_clsEmb_, d_lNormedClsEmb2_, d_d_normedCls2Output_, d_d_lNormedClsEmb2_,
                                    1, 1, embSize_);

                int mlpResidualSize = 1 * embSize_;
                d_mlpResidual_ = langHandle->allocDevMem((mlpResidualSize) * sizeof(float));
                d_d_mlpResidual_ = langHandle->allocDevMem((mlpResidualSize) * sizeof(float));
                

                int noOfLayers = 2;
                std::vector<int> layerOutputCounts;
                for(int i=0; i<noOfLayers; i++) {
                    layerOutputCounts.push_back(embSize_);
                }

                MlpIO mlpIO = {
                    1,                      // minibatchSize
                    embSize_,               // flaInputSize
                    embSize_,               // flaOutputSize
                    d_lNormedClsEmb2_,      // d_mlpInput
                    d_mlpResidual_,         // d_mlpOutput
                    d_d_lNormedClsEmb2_,    // d_mlpDx
                    d_d_mlpResidual_,       // d_mlpDy
                    2,                      // noOfLayers
                    layerOutputCounts       // layerOutputCount         
                }; 
                mlp_ = new Mlp(langHandle, timer, std::move(mlpIO));
                
                Tracer::func_end("CaitClassAttnStageEncoder::CaitClassAttnStageEncoder");    
            }

            void doFw() {
                Tracer::func_begin("CaitClassAttnStageEncoder::doFw");


                lNormPatchLayer_->doFw();
                lNormClsLayer1_->doFw();
                caitMultiHeadedClassAttn_->doFw();

                // add attention residual connection
                float scalingFactor = 1; 
                assertBlasInvar(BlasRoutines::doAxpy(langHandle_, 1 * embSize_, &scalingFactor, d_attnResidual_, d_clsEmb_));

                lNormClsLayer2_->doFw();
                mlp_->doFw();
                

                // add FFN residual connection
                assertBlasInvar(BlasRoutines::doAxpy(langHandle_, 1 * embSize_, &scalingFactor, d_mlpResidual_,  d_clsEmb_));

                Tracer::func_end("CaitClassAttnStageEncoder::doFw");    
            }

            void doBw() {
                Tracer::func_begin("CaitClassAttnStageEncoder::doBw");


                int copySize = 1 * embSize_;
                langHandle_->memCpyD2D(d_d_mlpResidual_, d_d_clsEmb_, sizeof(float) * copySize, true);

                mlp_->doBw(); 
                lNormClsLayer2_->doBw();

                float scalingFactor = 1; 
                int Z_size = 1 * embSize_;
                assertBlasInvar(BlasRoutines::doAxpy(langHandle_, Z_size, &scalingFactor, d_d_normedCls2Output_, d_d_clsEmb_));

                langHandle_->memCpyD2D(d_d_attnResidual_, d_d_clsEmb_, sizeof(float) * copySize, true);

                caitMultiHeadedClassAttn_->doBw();          
                lNormClsLayer1_->doBw();
                lNormPatchLayer_->doBw();



                Tracer::func_end("CaitClassAttnStageEncoder::doBw");      
                
            }

            ~CaitClassAttnStageEncoder() {
                Tracer::func_begin("CaitClassAttnStageEncoder::~CaitClassAttnStageEncoder");

                delete mlp_;
                delete caitMultiHeadedClassAttn_;
                delete lNormPatchLayer_;
                delete lNormClsLayer1_;
                delete lNormClsLayer2_;
                langHandle_->freeDevPtr(d_lNormedPatchEmbs_);
                langHandle_->freeDevPtr(d_d_lNormedPatchEmbs_);
                langHandle_->freeDevPtr(d_d_normedPatchOutput_);
                langHandle_->freeDevPtr(d_lNormedClsEmb1_);
                langHandle_->freeDevPtr(d_d_lNormedClsEmb1_);
                langHandle_->freeDevPtr(d_d_normedCls1Output_);
                langHandle_->freeDevPtr(d_attnResidual_);
                langHandle_->freeDevPtr(d_d_attnResidual_);
                langHandle_->freeDevPtr(d_lNormedClsEmb2_);
                langHandle_->freeDevPtr(d_d_lNormedClsEmb2_);
                langHandle_->freeDevPtr(d_d_normedCls2Output_);
                langHandle_->freeDevPtr(d_mlpResidual_);
                langHandle_->freeDevPtr(d_d_mlpResidual_);
                Tracer::func_end("CaitClassAttnStageEncoder::~CaitClassAttnStageEncoder");    
            }



    };

    class CaitClassAttentionStageEncoderController {
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




                CaitClassAttnStageEncoder *caitClassAttnStageEncoder = new CaitClassAttnStageEncoder(langHandle, timer, batchSize,
                                                        embSize, embByNoOfHeadsSize, noOfEmbs, noOfHeads, d_patchEmbs, d_d_patchEmbs,
                                                        d_clsEmb, d_d_clsEmb);




                int iterations = 2; 
                for(int i=0; i<iterations; i++) {
                    ImageProcessor::initImage(h_patchEmbs, patchEmbsSize);
                    ImageProcessor::initImage(h_clsEmb, clsEmbSize);
                    langHandle->memCpyH2D(d_patchEmbs, h_patchEmbs, sizeof(float) * patchEmbsSize, false);
                    langHandle->memCpyH2D(d_clsEmb, h_clsEmb, sizeof(float) * clsEmbSize, false);
                    langHandle->synchronize();
                    caitClassAttnStageEncoder->doFw();

                    ImageProcessor::initImage(h_d_patchEmbs, patchEmbsSize);
                    ImageProcessor::initImage(h_d_clsEmb, clsEmbSize);
                    langHandle->memCpyH2D(d_d_patchEmbs, h_d_patchEmbs, sizeof(float) * patchEmbsSize, false);
                    langHandle->memCpyH2D(d_d_clsEmb, h_d_clsEmb, sizeof(float) * clsEmbSize, false);
                    langHandle->synchronize();
                    caitClassAttnStageEncoder->doBw();
                }

            }
    };
};


#endif