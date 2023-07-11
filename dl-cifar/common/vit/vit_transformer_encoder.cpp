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

#include "vit_transformer_encoder.h"


namespace dl_cifar::common {
    VitTransformerEncoder::VitTransformerEncoder(LangHandle *langHandle, Timer* timer, int batchSize,
        int embSize, int intermediateEmbSize, int noOfEmbs, int noOfHeads, float *d_patchEmbs, float *d_d_patchEmbs)
        : langHandle_(langHandle), timer_(timer), batchSize_(batchSize), embSize_(embSize), 
            intermediateEmbSize_(intermediateEmbSize), noOfEmbs_(noOfEmbs), d_patchEmbs_(d_patchEmbs), d_d_patchEmbs_(d_d_patchEmbs)  {

        Tracer::func_begin("VitTransformerEncoder::VitTransformerEncoder");        
        


        // ------------- initialization for multi headed self attention --------------------------------------------
        int lNormedPatchEmbs1Size = batchSize * noOfEmbs * embSize;
        d_lNormedPatchEmbs1_ = langHandle->allocDevMem((lNormedPatchEmbs1Size) * sizeof(float));
        d_d_lNormedPatchEmbs1_ = langHandle->allocDevMem((lNormedPatchEmbs1Size) * sizeof(float));
        d_d_norm1Output_ = langHandle->allocDevMem((lNormedPatchEmbs1Size) * sizeof(float));

        int attnResidualSize = batchSize * noOfEmbs * embSize;
        d_attnResidual_ = langHandle->allocDevMem((attnResidualSize) * sizeof(float));
        d_d_attnResidual_ = langHandle->allocDevMem((attnResidualSize) * sizeof(float));

        lNormLayer1_ = new LNormLayer(langHandle, timer, d_patchEmbs_, d_lNormedPatchEmbs1_, d_d_norm1Output_, d_d_lNormedPatchEmbs1_,
                            batchSize_, noOfEmbs_, embSize_);


        multiHeadedAttention_ = new VitMultiHeadedSelfAttn(langHandle, timer, batchSize, embSize, intermediateEmbSize, 
                            noOfEmbs, noOfHeads, d_lNormedPatchEmbs1_, d_attnResidual_, d_d_lNormedPatchEmbs1_, d_d_attnResidual_);




        // ------------- initialization for FFN  --------------------------------------------

        int lNormedPatchEmbs2Size = batchSize * noOfEmbs * embSize;
        d_lNormedPatchEmbs2_ = langHandle->allocDevMem((lNormedPatchEmbs2Size) * sizeof(float));
        d_d_lNormedPatchEmbs2_ = langHandle->allocDevMem((lNormedPatchEmbs2Size) * sizeof(float));
        d_d_norm2Output_ = langHandle->allocDevMem((lNormedPatchEmbs2Size) * sizeof(float));


        int mlpResidualSize = batchSize * noOfEmbs * embSize;
        d_mlpResidual_ = langHandle->allocDevMem((mlpResidualSize) * sizeof(float));
        d_d_mlpResidual_ = langHandle->allocDevMem((mlpResidualSize) * sizeof(float));
        

        lNormLayer2_ = new LNormLayer(langHandle, timer, d_patchEmbs_, d_lNormedPatchEmbs2_, d_d_norm2Output_, d_d_lNormedPatchEmbs2_,
                            batchSize_, noOfEmbs_, embSize_);


        int noOfLayers = 2;
        std::vector<int> layerOutputCounts;
        for(int i=0; i<noOfLayers; i++) {
            layerOutputCounts.push_back(embSize_);
        }


        MlpIO mlpIO = {
            batchSize * noOfEmbs_,  // minibatchSize
            embSize_,               // flaInputSize
            embSize_,               // flaOutputSize
            d_lNormedPatchEmbs2_,   // d_mlpInput
            d_mlpResidual_,         // d_mlpOutput
            d_d_lNormedPatchEmbs2_, // d_mlpDx
            d_d_mlpResidual_,       // d_mlpDy
            2,                      // noOfLayers
            layerOutputCounts       // layerOutputCount         
        }; 
        mlp_ = new Mlp(langHandle, timer, std::move(mlpIO));
        
        Tracer::func_end("VitTransformerEncoder::VitTransformerEncoder");    
    }

    void VitTransformerEncoder::doFw() {
        Tracer::func_begin("VitTransformerEncoder::doFw");        
        
        
        lNormLayer1_->doFw();
        multiHeadedAttention_->doFw();

        // add residual connection
        float scalingFactor = 1; 
        int Z_size = batchSize_ * noOfEmbs_ * embSize_;
        assertBlasInvar(BlasRoutines::doAxpy(langHandle_, Z_size, &scalingFactor, d_attnResidual_, d_patchEmbs_));

        lNormLayer2_->doFw();
        mlp_->doFw();

        // add residual connection
        assertBlasInvar(BlasRoutines::doAxpy(langHandle_, Z_size, &scalingFactor, d_mlpResidual_, d_patchEmbs_));

        Tracer::func_end("VitTransformerEncoder::doFw");    
    }

    void VitTransformerEncoder::doBw() {
        Tracer::func_begin("VitTransformerEncoder::doBw");        
        
        // add residual connection
        // float scalingFactor = 1; 
        // int Z_size = batchSize_ * noOfEmbs_ * embSize_;
        // assertBlasInvar(BlasRoutines::doAxpy(langHandle_, Z_size, &scalingFactor, d_mlpResidual_, d_patchEmbs_));

        int copySize = batchSize_ * noOfEmbs_ * embSize_;
        langHandle_->memCpyD2D(d_d_mlpResidual_, d_d_patchEmbs_, sizeof(float) * copySize, true);

        mlp_->doBw(); 

        lNormLayer2_->doBw();

        float scalingFactor = 1; 
        int Z_size = batchSize_ * noOfEmbs_ * embSize_;
        assertBlasInvar(BlasRoutines::doAxpy(langHandle_, Z_size, &scalingFactor, d_d_norm2Output_, d_d_patchEmbs_));
        langHandle_->memCpyD2D(d_d_attnResidual_, d_d_patchEmbs_, sizeof(float) * copySize, true);

        multiHeadedAttention_->doBw();          
        lNormLayer1_->doBw();

        assertBlasInvar(BlasRoutines::doAxpy(langHandle_, Z_size, &scalingFactor, d_d_norm1Output_, d_d_patchEmbs_));
                    
        Tracer::func_end("VitTransformerEncoder::doBw");    
    }

    VitTransformerEncoder::~VitTransformerEncoder() {
        Tracer::func_begin("VitTransformerEncoder::~VitTransformerEncoder");        
        
        delete mlp_;
        delete multiHeadedAttention_;
        langHandle_->freeDevPtr(d_lNormedPatchEmbs1_);
        langHandle_->freeDevPtr(d_d_lNormedPatchEmbs1_);
        langHandle_->freeDevPtr(d_d_norm1Output_);
        langHandle_->freeDevPtr(d_attnResidual_);
        langHandle_->freeDevPtr(d_d_attnResidual_);
        langHandle_->freeDevPtr(d_lNormedPatchEmbs2_);
        langHandle_->freeDevPtr(d_d_lNormedPatchEmbs2_);
        langHandle_->freeDevPtr(d_d_norm2Output_);
        langHandle_->freeDevPtr(d_mlpResidual_);
        langHandle_->freeDevPtr(d_d_mlpResidual_);
        delete lNormLayer2_;
        delete lNormLayer1_;
        Tracer::func_end("VitTransformerEncoder::~VitTransformerEncoder");    
    }
};