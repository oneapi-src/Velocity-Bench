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

#include "vit_multi_headed_self_attn.h"
namespace dl_cifar::common {
    VitMultiHeadedSelfAttn::VitMultiHeadedSelfAttn(LangHandle *langHandle, Timer* timer, int batchSize, int embedSize, 
                    int intermediateEmbSize, int noOfEmbeds, int noOfHeads, float *d_X, float *d_Z, float *d_dX, float *d_dZ)
            : langHandle_(langHandle), timer_(timer), batchSize_(batchSize), embSize_(embedSize), intermediateEmbSize_(intermediateEmbSize),
                noOfEmbs_(noOfEmbeds), noOfHeads_(noOfHeads), d_X_(d_X), d_Z_(d_Z), d_dX_(d_dX), d_dZ_(d_dZ) {

        Tracer::func_begin("VitMultiHeadedSelfAttn::VitMultiHeadedSelfAttn");
        
        concatIntermediateZSize_ = batchSize * noOfEmbeds * (intermediateEmbSize * noOfHeads);
        d_concatIntermediateZ_ = langHandle->allocDevMem((concatIntermediateZSize_) * sizeof(float));
        d_d_concatIntermediateZ_ = langHandle->allocDevMem((concatIntermediateZSize_) * sizeof(float));

        oWeightsSize_ = (intermediateEmbSize * noOfHeads) * embedSize;
        h_oWeights_   = (float*)calloc(oWeightsSize_,   sizeof(float));  
        ImageProcessor::initImage(h_oWeights_, oWeightsSize_);
        d_oWeights_ = langHandle->allocDevMem((oWeightsSize_) * sizeof(float));
        d_d_oWeights_ = langHandle->allocDevMem((oWeightsSize_) * sizeof(float));
        langHandle->memCpyH2D(d_oWeights_, h_oWeights_, sizeof(float) * oWeightsSize_, true);

        for(int i=0; i<noOfHeads; i++) {
            float *d_intermediateZ, *d_d_intermediateZ;
            int intermediateZSize = batchSize * noOfEmbeds * intermediateEmbSize;
            d_intermediateZ = langHandle->allocDevMem((intermediateZSize) * sizeof(float));
            d_d_intermediateZ = langHandle->allocDevMem((intermediateZSize) * sizeof(float));
            d_intermediateZs_.push_back(d_intermediateZ);
            d_d_intermediateZs_.push_back(d_d_intermediateZ);
            heads_.push_back(new VitSelfAttnHead(langHandle, timer, batchSize, embedSize, intermediateEmbSize, 
                                                            noOfEmbeds, d_X, d_intermediateZ, d_dX, d_d_intermediateZ));
        }

        Tracer::func_end("VitMultiHeadedSelfAttn::VitMultiHeadedSelfAttn");    
    }

    void VitMultiHeadedSelfAttn::doFw() {

        Tracer::func_begin("VitMultiHeadedSelfAttn::doFw");
        
        for(int i=0; i<noOfHeads_; i++) {
            heads_[i]->doFw();
        }

        // concatenating intermediate Zs
        for(int batchCounter=0; batchCounter<batchSize_; batchCounter++) {
            for(int embCounter=0; embCounter<noOfEmbs_; embCounter++) {
                for(int headCounter=0; headCounter<noOfHeads_; headCounter++) {
                    langHandle_->memCpyD2D(d_concatIntermediateZ_ + (batchCounter * noOfEmbs_ * (intermediateEmbSize_ * noOfHeads_)) 
                                + (embCounter * intermediateEmbSize_ * noOfHeads_) + (headCounter*intermediateEmbSize_), 
                                d_intermediateZs_[headCounter] + (batchCounter * noOfEmbs_ * intermediateEmbSize_)
                                + (embCounter*intermediateEmbSize_),
                                sizeof(float) * intermediateEmbSize_, false);
                }
            }
        }
        langHandle_->synchronize();

        assertBlasInvar(BlasRoutines::doMatMul(langHandle_, batchSize_*noOfEmbs_, intermediateEmbSize_*noOfHeads_, embSize_, 
                                    d_concatIntermediateZ_, d_oWeights_, d_Z_));

        Tracer::func_end("VitMultiHeadedSelfAttn::doFw");    
    }

    void VitMultiHeadedSelfAttn::doBw() {
        Tracer::func_begin("VitMultiHeadedSelfAttn::doBw");
        

        for(int i=0; i<noOfHeads_; i++) {
            heads_[i]->doBw();
        }

        // concatenating intermediate Zs
        for(int batchCounter=0; batchCounter<batchSize_; batchCounter++) {
            for(int embCounter=0; embCounter<noOfEmbs_; embCounter++) {
                for(int headCounter=0; headCounter<noOfHeads_; headCounter++) {
                    langHandle_->memCpyD2D(d_d_concatIntermediateZ_ + (batchCounter * noOfEmbs_ * (intermediateEmbSize_ * noOfHeads_)) 
                                + (embCounter * intermediateEmbSize_ * noOfHeads_) + (headCounter*intermediateEmbSize_),
                                d_d_intermediateZs_[headCounter] + (batchCounter * noOfEmbs_ * intermediateEmbSize_)
                                + (embCounter*intermediateEmbSize_),
                                sizeof(float) * intermediateEmbSize_, false);    
                }
            }
        }
        langHandle_->synchronize();

        assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, batchSize_*noOfEmbs_, embSize_, intermediateEmbSize_ * noOfHeads_, 
                                                            d_dZ_, d_oWeights_, d_d_concatIntermediateZ_));
        assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, intermediateEmbSize_ * noOfHeads_, batchSize_*noOfEmbs_, embSize_, 
                                                            d_concatIntermediateZ_, d_dZ_, d_d_oWeights_));
        Tracer::func_end("VitMultiHeadedSelfAttn::doBw");    
    }


    VitMultiHeadedSelfAttn::~VitMultiHeadedSelfAttn() {
        Tracer::func_begin("VitMultiHeadedSelfAttn::~VitMultiHeadedSelfAttn");
        
        for(int i=0; i<noOfHeads_; i++) {
            langHandle_->freeDevPtr(d_intermediateZs_[i]);
            langHandle_->freeDevPtr(d_d_intermediateZs_[i]);
            delete heads_[i];
        }
        free(h_oWeights_);
        langHandle_->freeDevPtr(d_oWeights_);
        langHandle_->freeDevPtr(d_d_oWeights_);
        langHandle_->freeDevPtr(d_concatIntermediateZ_);
        langHandle_->freeDevPtr(d_d_concatIntermediateZ_);
        Tracer::func_end("VitMultiHeadedSelfAttn::~VitMultiHeadedSelfAttn");    
    }
};