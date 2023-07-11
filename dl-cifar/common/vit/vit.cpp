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

#include "vit.h"

namespace dl_cifar::common {
    Vit::Vit(LangHandle *langHandle, Timer* timer, VitParams vitParams, float *d_inputImgs)
        : langHandle_(langHandle), timer_(timer), vitParams_(vitParams), d_inputImgs_(d_inputImgs) {

        Tracer::func_begin("Vit::Vit");    
        
        imgsSize_ = vitParams.batchSize * vitParams.imgNoOfChannels * vitParams.imgWidth * vitParams.imgHeight;
        d_imgPatches_   = langHandle->allocDevMem((imgsSize_) * sizeof(float));
        d_d_imgPatches_ = langHandle->allocDevMem((imgsSize_) * sizeof(float));
        
        noOfPatches_   = (vitParams.imgWidth * vitParams.imgHeight) / (vitParams.patchSize * vitParams.patchSize);
        patchEmbsSize_ = vitParams.batchSize * noOfPatches_ * vitParams_.embSize;

        d_patchEmbs_   = langHandle->allocDevMem((patchEmbsSize_) * sizeof(float));
        d_d_patchEmbs_ = langHandle->allocDevMem((patchEmbsSize_) * sizeof(float));
        h_d_patchEmbs_ = (float*)calloc(patchEmbsSize_, sizeof(float));   

        // We have not added implementation of extra [class] embedding, because it adds only a single embedding 
        // to the overall count of embeddings. For ex., VIT L16 has (384*384)/(16*16) = 576 patch embeddings  
        //
        // Also, in the case of fine tuning, 2D interpolation of pre-trained position embeddings is done, before being used. 
        // This is outside the scope of this workload, and hence not added here in the implementation.  


        std::vector<int> layerOutputCounts;
        layerOutputCounts.push_back(vitParams_.embSize);

        MlpIO patchToD_mlp_IO = {
            vitParams_.batchSize * noOfPatches_,                                        // minibatchSize
            vitParams_.imgNoOfChannels * vitParams_.patchSize * vitParams_.patchSize,   // flaInputSize
            vitParams_.embSize,                                                         // flaOutputSize
            d_imgPatches_,                                                              // d_mlpInput
            d_patchEmbs_,                                                               // d_mlpOutput
            d_d_imgPatches_,                                                            // d_mlpDx
            d_d_patchEmbs_,                                                             // d_mlpDy
            1,                                                                          // noOfLayers
            layerOutputCounts                                                           // layerOutputCount         
        }; 
        patchToEmb_mlp_ = new Mlp(langHandle_, timer, std::move(patchToD_mlp_IO));


        transformer_ = new VitTransformer(langHandle_, timer, vitParams_.noOfEncoders, vitParams_.batchSize , 
                                            vitParams_.embSize, vitParams_.embSize/vitParams_.noOfHeads, noOfPatches_, 
                                            vitParams_.noOfHeads, d_patchEmbs_, d_d_patchEmbs_);

        Tracer::func_end("Vit::Vit");                               

    }

    void Vit::doFw() {
        Tracer::func_begin("Vit::doFw");    
        
        ImageProcessor::convertDevImgsToPatches(langHandle_, d_inputImgs_, d_imgPatches_, vitParams_.batchSize,
                    vitParams_.imgNoOfChannels, vitParams_.imgWidth, vitParams_.imgHeight, vitParams_.patchSize);

        patchToEmb_mlp_->doFw();
        transformer_->doFw();
        Tracer::func_end("Vit::doFw");    
    }

    void Vit::doBw() {
        Tracer::func_begin("Vit::doBw");    
        
        // we simulate the initialiation of backward propagation across the entire network, 
        // by feeding in a randomly initiated d_d_patchEmbs_. 
        ImageProcessor::initImage(h_d_patchEmbs_, patchEmbsSize_);
        langHandle_->memCpyH2D(d_d_patchEmbs_, h_d_patchEmbs_, sizeof(float) * patchEmbsSize_, true);

        transformer_->doBw();
        patchToEmb_mlp_->doBw();

        Tracer::func_end("Vit::doBw");      
    }

    Vit::~Vit() {
        Tracer::func_begin("Vit::~Vit");    
        
        langHandle_->freeDevPtr(d_patchEmbs_);
        langHandle_->freeDevPtr(d_imgPatches_);
        langHandle_->freeDevPtr(d_d_imgPatches_);
        langHandle_->freeDevPtr(d_d_patchEmbs_);

        delete patchToEmb_mlp_;
        delete transformer_;
        free( h_d_patchEmbs_);
        Tracer::func_end("Vit::~Vit");    
    }
};