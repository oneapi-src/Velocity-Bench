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

#include "cait_class_attn_head.h"

namespace dl_cifar::common {
    CaitClassAttnHead::CaitClassAttnHead(LangHandle *langHandle, Timer* timer, size_t batchSize, size_t embSize, 
                            size_t intermediateEmbSize, size_t noOfEmbs, const float * const d_inputPatchEmbs, float *d_d_inputPatchEmbs,
                            float *d_inputClsEmb, float *d_d_inputClsEmb, float *d_outputClsEmbPerHead, float *d_d_outputClsEmbPerHead)
            : langHandle_(langHandle), timer_(timer), batchSize_(batchSize), embSize_(embSize), noOfEmbs_(noOfEmbs),
                intermediateEmbSize_(intermediateEmbSize), d_inputPatchEmbs_(d_inputPatchEmbs),d_inputClsEmb_(d_inputClsEmb), 
                d_outputClsEmbPerHead_(d_outputClsEmbPerHead), d_d_inputPatchEmbs_(d_d_inputPatchEmbs), d_d_inputClsEmb_(d_d_inputClsEmb), 
                d_d_outputClsEmbPerHead_(d_d_outputClsEmbPerHead) {

        Tracer::func_begin("CaitClassAttnHead::CaitClassAttnHead");

        //---------- create QKV weights -----------------------

        

        qkvWeightsSize_ = embSize * intermediateEmbSize;
        clsEmbPlusPatchEmbSize_ = batchSize_ * ((1 + noOfEmbs_) * embSize_);

        h_qWeights_   = (float*)calloc(qkvWeightsSize_,   sizeof(float));  
        h_kWeights_   = (float*)calloc(qkvWeightsSize_,   sizeof(float));  
        h_vWeights_   = (float*)calloc(qkvWeightsSize_,   sizeof(float));  

        ImageProcessor::initImage(h_qWeights_, qkvWeightsSize_);
        ImageProcessor::initImage(h_kWeights_, qkvWeightsSize_);
        ImageProcessor::initImage(h_vWeights_, qkvWeightsSize_);

        d_qWeights_ = langHandle->allocDevMem((qkvWeightsSize_) * sizeof(float));
        d_kWeights_ = langHandle->allocDevMem((qkvWeightsSize_) * sizeof(float));
        d_vWeights_ = langHandle->allocDevMem((qkvWeightsSize_) * sizeof(float));


        langHandle->memCpyH2D(d_qWeights_, h_qWeights_, sizeof(float) * qkvWeightsSize_, false);
        langHandle->memCpyH2D(d_kWeights_, h_kWeights_, sizeof(float) * qkvWeightsSize_, false);
        langHandle->memCpyH2D(d_vWeights_, h_vWeights_, sizeof(float) * qkvWeightsSize_, false);
        langHandle->synchronize();


        d_clsEmbPlusPatchEmbs_ = langHandle->allocDevMem((clsEmbPlusPatchEmbSize_) * sizeof(float));
        d_d_clsEmbPlusPatchEmbs_ = langHandle->allocDevMem((clsEmbPlusPatchEmbSize_) * sizeof(float));

        for( int i=0; i<batchSize_; i++) {             
            langHandle->memCpyD2D(d_clsEmbPlusPatchEmbs_ + (i * ((1 + noOfEmbs_) * embSize_)), d_inputClsEmb_ + (i * (1 * embSize_)), 
                                                                                            sizeof(float) * embSize_, false);
            langHandle->memCpyD2D(d_clsEmbPlusPatchEmbs_ + (i * ((1 + noOfEmbs_) * embSize_)) + embSize_, d_inputPatchEmbs_ + (i * (noOfEmbs_ * embSize_)), 
                                                                                            sizeof(float) * noOfEmbs_ * embSize_, false);
        }

        d_d_inputClsEmb_Vpath_ = langHandle->allocDevMem((batchSize_ * (1 + noOfEmbs_)* embSize_) * sizeof(float));

        langHandle->synchronize();
        
        d_d_qWeights_ = langHandle->allocDevMem((qkvWeightsSize_) * sizeof(float));
        d_d_kWeights_ = langHandle->allocDevMem((qkvWeightsSize_) * sizeof(float));
        d_d_vWeights_ = langHandle->allocDevMem((qkvWeightsSize_) * sizeof(float));
        

        //---------- create Q, K, V, QKt, Z -----------------------
        qSize_     = batchSize_ * 1 * intermediateEmbSize;
        kAndVSize_ = batchSize_ * ((1 + noOfEmbs) * intermediateEmbSize);

        d_Q_ = langHandle->allocDevMem((qSize_) * sizeof(float));
        d_K_ = langHandle->allocDevMem((kAndVSize_) * sizeof(float));
        d_V_ = langHandle->allocDevMem((kAndVSize_) * sizeof(float));

        d_dQ_ = langHandle->allocDevMem((qSize_) * sizeof(float));
        d_dK_ = langHandle->allocDevMem((kAndVSize_) * sizeof(float));
        d_dV_ = langHandle->allocDevMem((kAndVSize_) * sizeof(float));

        QKt_size_ = batchSize_ * (1 * (1 + noOfEmbs));
        d_QKt_ = langHandle->allocDevMem((QKt_size_) * sizeof(float));
        d_d_QKt_ = langHandle->allocDevMem((QKt_size_) * sizeof(float));

        scale_ = std::sqrt(intermediateEmbSize);
        scale_bw_ = 1/scale_;

        int inputTensorDims[4]   = {static_cast<int>(batchSize_), 1, static_cast<int>(1 + noOfEmbs), 1};
        int outputTensorDims[4]   = {static_cast<int>(batchSize_), 1, static_cast<int>(1 + noOfEmbs), 1};

        softmaxLayer_ = new SoftmaxLayer(langHandle, timer, inputTensorDims, outputTensorDims, d_QKt_, d_d_QKt_, d_QKt_, d_d_QKt_);

        Tracer::func_end("CaitClassAttnHead::CaitClassAttnHead");    
    }        

    CaitClassAttnHead::~CaitClassAttnHead() {
        Tracer::func_begin("CaitClassAttnHead::~CaitClassAttnHead");
        free(h_qWeights_);
        free(h_kWeights_);
        free(h_vWeights_);

        langHandle_->freeDevPtr(d_qWeights_);
        langHandle_->freeDevPtr(d_kWeights_);
        langHandle_->freeDevPtr(d_vWeights_);
        langHandle_->freeDevPtr(d_clsEmbPlusPatchEmbs_);
        langHandle_->freeDevPtr(d_d_qWeights_);
        langHandle_->freeDevPtr(d_d_kWeights_);
        langHandle_->freeDevPtr(d_d_vWeights_);
        langHandle_->freeDevPtr(d_Q_);
        langHandle_->freeDevPtr(d_K_);
        langHandle_->freeDevPtr(d_V_);
        langHandle_->freeDevPtr(d_dQ_);
        langHandle_->freeDevPtr(d_dK_);
        langHandle_->freeDevPtr(d_dV_);
        langHandle_->freeDevPtr(d_QKt_);
        delete softmaxLayer_;
        
        Tracer::func_end("CaitClassAttnHead::~CaitClassAttnHead");    
    }

    void CaitClassAttnHead::doFw() {
        Tracer::func_begin("CaitClassAttnHead::doFw");

        size_t clsEmbPlusPatchEmbsStride   = (1 + noOfEmbs_) * embSize_;
        size_t qStride   = 1  * intermediateEmbSize_;
        size_t kvStride  = (1 + noOfEmbs_) * intermediateEmbSize_;
        size_t qkTStride = 1 * (1 + noOfEmbs_);
        size_t outputClsEmbPerHeadStride = 1 * intermediateEmbSize_;
        
        for( int i=0; i<batchSize_; i++) {                
            assertBlasInvar(BlasRoutines::doMatMul(langHandle_, 1, embSize_, intermediateEmbSize_, 
                                d_inputClsEmb_ + (i * embSize_), d_qWeights_, d_Q_ + (i * qStride)));

            assertBlasInvar(BlasRoutines::doMatMul(langHandle_, 1 + noOfEmbs_, embSize_, intermediateEmbSize_, 
                                d_clsEmbPlusPatchEmbs_ + (i * clsEmbPlusPatchEmbsStride), d_kWeights_, d_K_ + (i * kvStride)));
            assertBlasInvar(BlasRoutines::doMatMul(langHandle_, 1 + noOfEmbs_, embSize_, intermediateEmbSize_, 
                                d_clsEmbPlusPatchEmbs_ + (i * clsEmbPlusPatchEmbsStride), d_vWeights_, d_V_ + (i * kvStride)));

            assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, 1, intermediateEmbSize_, 1 + noOfEmbs_, 
                                d_Q_, d_K_ + (i*kvStride), d_QKt_ + (i*qkTStride)));
        }

        assertBlasInvar(BlasRoutines::scaleVector(langHandle_, QKt_size_, &scale_, d_QKt_, 1));
        
        softmaxLayer_->doFw();

        for( int i=0; i<batchSize_; i++) {
            assertBlasInvar(BlasRoutines::doMatMul(langHandle_, 1, 1 + noOfEmbs_, intermediateEmbSize_, 
                d_QKt_ + (i * qkTStride), d_V_ + (i * kvStride), d_outputClsEmbPerHead_ + (i *  outputClsEmbPerHeadStride)));
        }

        Tracer::func_end("CaitClassAttnHead::doFw");    
    }

    void CaitClassAttnHead::doBw() {
        Tracer::func_begin("CaitClassAttnHead::doBw");

        size_t clsEmbPlusPatchEmbsStride   = (1 + noOfEmbs_) * embSize_;
        size_t qStride   = 1  * intermediateEmbSize_;
        size_t kvStride  = (1 + noOfEmbs_) * intermediateEmbSize_;
        size_t qkTStride = 1 * (1 + noOfEmbs_);
        size_t outputClsEmbPerHeadStride = 1 * intermediateEmbSize_;


        for( int i=0; i<batchSize_; i++) {
            assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, 1, intermediateEmbSize_,  (1 + noOfEmbs_), 
                            d_d_outputClsEmbPerHead_ + (i*outputClsEmbPerHeadStride), d_V_ + (i*kvStride), d_d_QKt_ + (i*qkTStride)));
            assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, (1 + noOfEmbs_), 1, intermediateEmbSize_, 
                            d_QKt_ + (i*qkTStride), d_d_outputClsEmbPerHead_ + (i*outputClsEmbPerHeadStride), d_dV_ + (i*kvStride)));
        }

        softmaxLayer_->doFw();

        assertBlasInvar(BlasRoutines::scaleVector(langHandle_, QKt_size_, &scale_bw_, d_d_QKt_, 1));

        for( int i=0; i<batchSize_; i++) {
            assertBlasInvar(BlasRoutines::doMatMul(langHandle_, 1, (1 + noOfEmbs_), intermediateEmbSize_, 
                                        d_d_QKt_ + (i*qkTStride), d_K_ + (i*kvStride), d_dQ_ + (i*qStride)));
            assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, (1 + noOfEmbs_), 1, intermediateEmbSize_, 
                                        d_d_QKt_ + (i*qkTStride), d_Q_ + (i*qStride), d_dK_ + (i*kvStride)));

        }


        assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, batchSize_ * 1, intermediateEmbSize_, embSize_,
                                        d_dQ_, d_qWeights_, d_d_inputClsEmb_));
        assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, embSize_, batchSize_* 1, intermediateEmbSize_, 
                                        d_inputClsEmb_, d_dQ_, d_d_qWeights_));
        
        assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, batchSize_ * (1 + noOfEmbs_), intermediateEmbSize_, embSize_, 
                                        d_dK_, d_kWeights_, d_d_inputPatchEmbs_));
        assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, embSize_, batchSize_ * (1 + noOfEmbs_), intermediateEmbSize_, 
                                        d_clsEmbPlusPatchEmbs_, d_dK_, d_d_kWeights_));

        assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, batchSize_ * (1 + noOfEmbs_), intermediateEmbSize_, embSize_, 
                                        d_dV_, d_vWeights_, d_d_inputClsEmb_Vpath_));
        assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, embSize_, batchSize_ * (1 + noOfEmbs_), intermediateEmbSize_, 
                                        d_clsEmbPlusPatchEmbs_, d_dV_, d_d_vWeights_));

        float scalingFactor = 1; 
        int inputClsEmbSize = batchSize_ * (1 + noOfEmbs_) * embSize_;
        assertBlasInvar(BlasRoutines::doAxpy(langHandle_, inputClsEmbSize, &scalingFactor, d_d_inputClsEmb_Vpath_, d_d_inputPatchEmbs_));

        Tracer::func_end("CaitClassAttnHead::doBw");        
    }
};