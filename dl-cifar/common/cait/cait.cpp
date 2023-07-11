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

#include "cait.h"

namespace dl_cifar::common {
    Cait::Cait(LangHandle *langHandle, Timer* timer, CaitParams caitParams, float *d_inputImgs)
        : langHandle_(langHandle), timer_(timer), noOfSAEncoders_(caitParams.noOfSAEncoders), noOfCAEncoders_(caitParams.noOfCAEncoders), 
            caitParams_(caitParams), d_inputImgs_(d_inputImgs) {

        Tracer::func_begin("Cait::Cait");

        imgsSize_ = caitParams.batchSize * caitParams.imgNoOfChannels * caitParams.imgWidth * caitParams.imgHeight;
        d_imgPatches_ = langHandle->allocDevMem((imgsSize_) * sizeof(float));
        d_d_imgPatches_ = langHandle->allocDevMem((imgsSize_) * sizeof(float));
        
        noOfPatches_ = (caitParams.imgWidth * caitParams.imgHeight) / (caitParams.patchSize * caitParams.patchSize);  
        patchEmbsSize_ = caitParams.batchSize * noOfPatches_ * caitParams.embSize;    

        d_patchEmbs_ = langHandle->allocDevMem((patchEmbsSize_) * sizeof(float));
        h_d_patchEmbs_   = (float*)calloc(patchEmbsSize_,   sizeof(float));  
        d_d_patchEmbs_ = langHandle->allocDevMem((patchEmbsSize_) * sizeof(float));

        //---------- create class embedding -------------------------------
        clsEmbSize_ = caitParams.batchSize * caitParams.embSize;
        h_d_clsEmb_   = (float*)calloc(clsEmbSize_,   sizeof(float));  
        d_clsEmb_ = langHandle->allocDevMem((clsEmbSize_) * sizeof(float));
        d_d_clsEmb_ = langHandle->allocDevMem((clsEmbSize_) * sizeof(float));
        //-------------------------------------------------------------------


        std::vector<int> layerOutputCounts;
        layerOutputCounts.push_back(caitParams.embSize);

        MlpIO patchToD_mlp_IO = {
            caitParams.batchSize * noOfPatches_,                                        // minibatchSize
            caitParams.imgNoOfChannels * caitParams.patchSize * caitParams.patchSize,   // flaInputSize
            caitParams.embSize,                                                         // flaOutputSize
            d_imgPatches_,                                                              // d_mlpInput
            d_patchEmbs_,                                                               // d_mlpOutput
            d_d_imgPatches_,                                                            // d_mlpDx
            d_d_patchEmbs_,                                                             // d_mlpDy
            1,                                                                          // noOfLayers
            layerOutputCounts                                                           // layerOutputCount         
        }; 
        patchToEmb_mlp_ = new Mlp(langHandle, timer, std::move(patchToD_mlp_IO));




        selfAttentionStage_  = new CaitSelfAttnStage(langHandle, timer, caitParams.noOfSAEncoders, caitParams.batchSize,
            caitParams.embSize, caitParams.embSize/caitParams.noOfHeads, caitParams.batchSize, caitParams.noOfHeads,
            d_patchEmbs_, d_d_patchEmbs_);

        classAttentionStage_ = new CaitClassAttnStage(langHandle, timer, caitParams.noOfCAEncoders, caitParams.batchSize,
            caitParams.embSize, caitParams.embSize/caitParams.noOfHeads, caitParams.batchSize, caitParams.noOfHeads, d_patchEmbs_, 
            d_d_patchEmbs_, d_clsEmb_, d_d_clsEmb_);


        Tracer::func_end("Cait::Cait");     
    }

    void Cait::doFw() {
        Tracer::func_begin("Cait::doFw");
        ImageProcessor::convertDevImgsToPatches(langHandle_, d_inputImgs_, d_imgPatches_, caitParams_.batchSize,
                    caitParams_.imgNoOfChannels, caitParams_.imgWidth, caitParams_.imgHeight, caitParams_.patchSize);

        patchToEmb_mlp_->doFw();
        selfAttentionStage_->doFw();
        classAttentionStage_->doFw();

        Tracer::func_end("Cait::doFw");    
    }

    void Cait::doBw() {
        Tracer::func_begin("Cait::doBw");

        // we simulate the initialiation of backward propagation across the entire network, 
        // by feeding in a randomly initiated d_d_patchEmbs_ and d_d_clsEmb_. 
        ImageProcessor::initImage(h_d_patchEmbs_, patchEmbsSize_);
        langHandle_->memCpyH2D(d_d_patchEmbs_, h_d_patchEmbs_, sizeof(float) * patchEmbsSize_, false);
        ImageProcessor::initImage(h_d_clsEmb_, clsEmbSize_);
        langHandle_->memCpyH2D(d_d_clsEmb_, h_d_clsEmb_, sizeof(float) * clsEmbSize_, false);
        langHandle_->synchronize();

        classAttentionStage_->doBw();
        selfAttentionStage_->doBw();
        patchToEmb_mlp_->doBw();
        

        Tracer::func_end("Cait::doBw");    
    }



    Cait::~Cait() {
        Tracer::func_begin("Cait::~Cait");
        
        delete selfAttentionStage_;
        delete classAttentionStage_;
        delete patchToEmb_mlp_;
        free(h_d_patchEmbs_);
        free(h_d_clsEmb_);
        langHandle_->freeDevPtr(d_imgPatches_);
        langHandle_->freeDevPtr(d_d_imgPatches_);

        Tracer::func_end("Cait::~Cait");    
    }
};