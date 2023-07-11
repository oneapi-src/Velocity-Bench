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

#include "vit_self_attn_head.h"
namespace dl_cifar::common {
    VitSelfAttnHead::VitSelfAttnHead(LangHandle *langHandle, Timer* timer, int batchSize, int embSize, 
                            int intermediateEmbSize, int noOfEmbs, float *d_X, float *d_intermediateZ, float *d_dX, float *d_d_intermediateZ)
            : langHandle_(langHandle), timer_(timer), batchSize_(batchSize), embSize_(embSize), 
                intermediateEmbSize_(intermediateEmbSize), noOfEmbs_(noOfEmbs), d_X_(d_X), d_intermediateZ_(d_intermediateZ), 
                d_dX_(d_dX), d_d_intermediateZ_(d_d_intermediateZ) {


        Tracer::func_begin("VitSelfAttnHead::VitSelfAttnHead");     

        //---------- create QKV weights -----------------------
        qkvWeightsSize_ = embSize_ * intermediateEmbSize_;

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

        d_d_qWeights_ = langHandle->allocDevMem((qkvWeightsSize_) * sizeof(float));
        d_d_kWeights_ = langHandle->allocDevMem((qkvWeightsSize_) * sizeof(float));
        d_d_vWeights_ = langHandle->allocDevMem((qkvWeightsSize_) * sizeof(float));

        d_dX_Kpath_ = langHandle->allocDevMem((batchSize_ * noOfEmbs_ * embSize_) * sizeof(float));
        d_dX_Vpath_ = langHandle->allocDevMem((batchSize_ * noOfEmbs_ * embSize_) * sizeof(float));


        //---------- create Q, K, V, QKt, Z -----------------------
        qkvSize_ = batchSize_ * noOfEmbs_ * intermediateEmbSize_;

        d_Q_ = langHandle->allocDevMem((qkvSize_) * sizeof(float));
        d_K_ = langHandle->allocDevMem((qkvSize_) * sizeof(float));
        d_V_ = langHandle->allocDevMem((qkvSize_) * sizeof(float));


        d_dQ_ = langHandle->allocDevMem((qkvSize_) * sizeof(float));
        d_dK_ = langHandle->allocDevMem((qkvSize_) * sizeof(float));
        d_dV_ = langHandle->allocDevMem((qkvSize_) * sizeof(float));

        QKt_size_ = batchSize_ * noOfEmbs_ * noOfEmbs_;
        d_QKt_ = langHandle->allocDevMem((QKt_size_) * sizeof(float));
        d_d_QKt_ = langHandle->allocDevMem((QKt_size_) * sizeof(float));

        scale_ = std::sqrt(intermediateEmbSize_);
        scale_bw_ = 1/scale_;

        int inputTensorDims[4]  = {batchSize_, noOfEmbs_, noOfEmbs_, 1};
        int outputTensorDims[4] = {batchSize_, noOfEmbs_, noOfEmbs_, 1};
        
        softmaxLayer_ = new SoftmaxLayer(langHandle, timer, inputTensorDims, outputTensorDims, d_QKt_, d_d_QKt_, d_QKt_, d_d_QKt_);

        Tracer::func_end("VitSelfAttnHead::VitSelfAttnHead");    
    }        

    VitSelfAttnHead::~VitSelfAttnHead() {
        Tracer::func_begin("VitSelfAttnHead::~VitSelfAttnHead");     

        langHandle_->freeDevPtr(d_qWeights_);
        langHandle_->freeDevPtr(d_kWeights_);
        langHandle_->freeDevPtr(d_vWeights_);
        free(h_qWeights_);
        free(h_kWeights_);
        free(h_vWeights_);

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
        langHandle_->freeDevPtr(d_d_QKt_);

        langHandle_->freeDevPtr(d_dX_Vpath_);
        langHandle_->freeDevPtr(d_dX_Kpath_);

        delete softmaxLayer_;
        
        Tracer::func_end("VitSelfAttnHead::~VitSelfAttnHead");    

    }

    void VitSelfAttnHead::doFw() {
        Tracer::func_begin("VitSelfAttnHead::doFw");     
        int xSize          = noOfEmbs_  * embSize_; 
        int qkvWeightsSize = embSize_   * intermediateEmbSize_;
        int qkvSize        = noOfEmbs_  * intermediateEmbSize_;

        int xStride   = noOfEmbs_ * embSize_;
        int qkvStride = noOfEmbs_ * intermediateEmbSize_;
        int qkTStride = noOfEmbs_ * noOfEmbs_;
        int zStride   = noOfEmbs_*intermediateEmbSize_;

        for( int i=0; i<batchSize_; i++) {
            assertBlasInvar(BlasRoutines::doMatMul(langHandle_, noOfEmbs_, intermediateEmbSize_, intermediateEmbSize_, 
                                        d_X_ + (i*xStride), d_qWeights_, d_Q_ + (i*qkvStride)));
            assertBlasInvar(BlasRoutines::doMatMul(langHandle_, noOfEmbs_, intermediateEmbSize_, intermediateEmbSize_, 
                                        d_X_ + (i*xStride), d_kWeights_, d_K_ + (i*qkvStride)));
            assertBlasInvar(BlasRoutines::doMatMul(langHandle_, noOfEmbs_, intermediateEmbSize_, intermediateEmbSize_, 
                                        d_X_ + (i*xStride), d_vWeights_, d_V_ + (i*qkvStride)));

            assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, noOfEmbs_, intermediateEmbSize_, noOfEmbs_,
                                        d_Q_ + (i*qkvStride), d_K_ + (i*qkvStride), d_QKt_ + (i*qkTStride)));
        }

        assertBlasInvar(BlasRoutines::scaleVector(langHandle_, QKt_size_, &scale_, d_QKt_, 1));

        softmaxLayer_->doFw();

        for( int i=0; i<batchSize_; i++) {
            assertBlasInvar(BlasRoutines::doMatMul(langHandle_, noOfEmbs_, noOfEmbs_, intermediateEmbSize_,
                                        d_QKt_ + (i*qkTStride), d_V_ + (i*qkvStride), d_intermediateZ_ + (i*zStride)));
        }

        Tracer::func_end("VitSelfAttnHead::doFw");    
    }

    void VitSelfAttnHead::doBw() {
        Tracer::func_begin("VitSelfAttnHead::doBw");     

        int xStride   = noOfEmbs_ * embSize_;
        int qkvStride = noOfEmbs_ * intermediateEmbSize_;
        int qkTStride = noOfEmbs_ * noOfEmbs_;
        int zStride   = noOfEmbs_ * intermediateEmbSize_;


        for( int i=0; i<batchSize_; i++) {
            assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, noOfEmbs_, intermediateEmbSize_, noOfEmbs_, 
                                        d_d_intermediateZ_ + (i*zStride), d_V_ + (i*qkvStride), d_d_QKt_ + (i*qkTStride)));
            assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, noOfEmbs_, noOfEmbs_, intermediateEmbSize_, 
                                        d_QKt_ + (i*qkTStride), d_d_intermediateZ_ + (i*zStride), d_dV_ + (i*qkvStride)));

        }
        
        softmaxLayer_->doBw();

        assertBlasInvar(BlasRoutines::scaleVector(langHandle_, QKt_size_, &scale_bw_, d_d_QKt_, 1));


        for( int i=0; i<batchSize_; i++) {
            assertBlasInvar(BlasRoutines::doMatMul(langHandle_, noOfEmbs_, noOfEmbs_, intermediateEmbSize_, 
                                        d_d_QKt_ + (i*qkTStride), d_K_ + (i*qkvStride), d_dQ_ + (i*qkvStride)));
            assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, noOfEmbs_, noOfEmbs_, intermediateEmbSize_, 
                                        d_d_QKt_ + (i*qkTStride), d_Q_ + (i*qkvStride), d_dK_ + (i*qkvStride)));

        }

        assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, batchSize_*noOfEmbs_, intermediateEmbSize_, embSize_,
                                        d_dQ_, d_qWeights_, d_dX_));
        assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, embSize_, batchSize_*noOfEmbs_, intermediateEmbSize_, 
                                        d_X_, d_dQ_, d_d_qWeights_));

        assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, batchSize_*noOfEmbs_, intermediateEmbSize_, embSize_, 
                                        d_dK_, d_kWeights_, d_dX_Kpath_));
        assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, embSize_, batchSize_*noOfEmbs_, intermediateEmbSize_, 
                                        d_X_, d_dK_, d_d_kWeights_));

        assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, batchSize_*noOfEmbs_, intermediateEmbSize_, embSize_, 
                                        d_dV_, d_vWeights_, d_dX_Vpath_));
        assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, embSize_, batchSize_*noOfEmbs_, intermediateEmbSize_, 
                                        d_X_, d_dV_, d_d_vWeights_));

        float scalingFactor = 1; 
        int xSize = batchSize_ * noOfEmbs_ * embSize_;
        assertBlasInvar(BlasRoutines::doAxpy(langHandle_, xSize, &scalingFactor, d_dX_Kpath_, d_dX_));
        assertBlasInvar(BlasRoutines::doAxpy(langHandle_, xSize, &scalingFactor, d_dX_Vpath_, d_dX_));

            

        Tracer::func_end("VitSelfAttnHead::doBw");    
    }
};