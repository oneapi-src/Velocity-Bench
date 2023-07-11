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

#include "cait_multi_headed_class_attn.h"

namespace dl_cifar::common {
CaitMultiHeadedClassAttn::CaitMultiHeadedClassAttn(LangHandle *langHandle, Timer* timer, int batchSize, int embSize, int intermediateEmbSize, int noOfEmbs,
                int noOfHeads, const float * const d_inputPatchEmbs, float *d_d_inputPatchEmbs, float *d_inputClsEmb, float *d_d_inputClsEmb,
                    float *d_outputAttnResiual, float *d_d_outputAttnResiual)
        : langHandle_(langHandle), batchSize_(batchSize), timer_(timer), embSize_(embSize), intermediateEmbSize_(intermediateEmbSize), noOfEmbs_(noOfEmbs), 
        noOfHeads_(noOfHeads), d_inputPatchEmbs_(d_inputPatchEmbs), d_d_inputPatchEmbs_(d_d_inputPatchEmbs), d_inputClsEmb_(d_inputClsEmb), 
        d_d_inputClsEmb_(d_d_inputClsEmb), d_outputAttnResiual_(d_outputAttnResiual), d_d_outputAttnResiual_(d_d_outputAttnResiual) {


    Tracer::func_begin("CaitMultiHeadedClassAttn::CaitMultiHeadedClassAttn");
    assert (intermediateEmbSize * noOfHeads == embSize);

    concatIntermediateZSize_ = batchSize_ * intermediateEmbSize * noOfHeads;
    d_concatIntermediateZ_ = langHandle->allocDevMem((concatIntermediateZSize_) * sizeof(float));
    d_d_concatIntermediateZ_ = langHandle->allocDevMem((concatIntermediateZSize_) * sizeof(float));

    oWeightsSize_ = (intermediateEmbSize * noOfHeads) * embSize;
    h_oWeights_   = (float*)calloc(oWeightsSize_,   sizeof(float));  
    ImageProcessor::initImage(h_oWeights_, oWeightsSize_);
    d_oWeights_ = langHandle->allocDevMem((oWeightsSize_) * sizeof(float));
    langHandle->memCpyH2D(d_oWeights_, h_oWeights_, sizeof(float) * oWeightsSize_, true);

    for(int i=0; i<noOfHeads; i++) {
        float *d_intermediateZ, *d_d_intermediateZ;
        int intermediateZSize = batchSize_ * intermediateEmbSize;
        d_intermediateZ = langHandle->allocDevMem((intermediateZSize) * sizeof(float));
        d_d_intermediateZ = langHandle->allocDevMem((intermediateZSize) * sizeof(float));
        d_intermediateZs_.push_back(d_intermediateZ);
        d_d_intermediateZs_.push_back(d_d_intermediateZ);
        heads_.push_back(new CaitClassAttnHead(langHandle, timer, batchSize, embSize, intermediateEmbSize, 
                                            noOfEmbs, d_inputPatchEmbs, d_d_inputPatchEmbs_, d_inputClsEmb, d_d_inputClsEmb_,
                                            d_intermediateZ, d_d_intermediateZ));
    }
    Tracer::func_end("CaitMultiHeadedClassAttn::CaitMultiHeadedClassAttn");   
}

void CaitMultiHeadedClassAttn::doFw() {

    Tracer::func_begin("CaitMultiHeadedClassAttn::doFw");
    for(int i=0; i<noOfHeads_; i++) {
        heads_[i]->doFw();
    }

    // concatenating individual z
    for (int headCounter=0; headCounter<noOfHeads_; headCounter++) {
        langHandle_->memCpyD2D(d_concatIntermediateZ_ + (headCounter*intermediateEmbSize_),
                d_intermediateZs_[headCounter], sizeof(float) * intermediateEmbSize_, false);
    }
    langHandle_->synchronize();

    
    assertBlasInvar(BlasRoutines::doMatMul(langHandle_, 1, embSize_, embSize_, 
                                d_concatIntermediateZ_, d_oWeights_, d_outputAttnResiual_));

    Tracer::func_end("CaitMultiHeadedClassAttn::doFw");   
}

void CaitMultiHeadedClassAttn::doBw() {
    Tracer::func_begin("CaitMultiHeadedClassAttn::doBw");
    for(int i=0; i<noOfHeads_; i++) {
        heads_[i]->doBw();
    }

        // concatenating individual z
    for (int headCounter=0; headCounter<noOfHeads_; headCounter++) {
        langHandle_->memCpyD2D(d_d_concatIntermediateZ_ + (headCounter*intermediateEmbSize_), 
                d_d_intermediateZs_[headCounter], sizeof(float) * intermediateEmbSize_, false);
    }
    langHandle_->synchronize();


    Tracer::func_end("CaitMultiHeadedClassAttn::doBw");   
}

CaitMultiHeadedClassAttn::~CaitMultiHeadedClassAttn() {
    Tracer::func_begin("CaitMultiHeadedClassAttn::~CaitMultiHeadedClassAttn");
    
    for(int i=0; i<noOfHeads_; i++) {
        langHandle_->freeDevPtr(d_intermediateZs_[i]);
        langHandle_->freeDevPtr(d_d_intermediateZs_[i]);
        delete heads_[i];
    }
    free(h_oWeights_);
    langHandle_->freeDevPtr(d_oWeights_);
    langHandle_->freeDevPtr(d_concatIntermediateZ_);
    langHandle_->freeDevPtr(d_d_concatIntermediateZ_);
    Tracer::func_end("CaitMultiHeadedClassAttn::~CaitMultiHeadedClassAttn");   
}
};