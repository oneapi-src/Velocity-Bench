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

#include "mlp.h"

namespace dl_cifar::common {
        
    LinearLayer::LinearLayer(LangHandle *langHandle, Timer* timer, int minibatchSize, int flattenedInputSize, int flattenedOutputSize, 
                                                    float *d_input, float *d_output, float *d_dx, float *d_dy)
        : langHandle_(langHandle), timer_(timer), minibatchSize_(minibatchSize), 
            flattenedInputSize_(flattenedInputSize), flattenedOutputSize_(flattenedOutputSize), 
            d_input_(d_input), d_output_(d_output), d_dx_(d_dx), d_dy_(d_dy) {

        Tracer::func_begin("LinearLayer::LinearLayer");

        inputSize_   = minibatchSize * flattenedInputSize;
        weightsSize_ = flattenedInputSize * flattenedOutputSize;
        outputSize_  = minibatchSize * flattenedOutputSize;

        //create and initialize weights
        h_weights_ = (float*)calloc(weightsSize_, sizeof(float));      
        ImageProcessor::initImage(h_weights_, weightsSize_);    
        d_weights_ = langHandle->allocDevMem((weightsSize_) * sizeof(float));
        langHandle->memCpyH2D(d_weights_, h_weights_, sizeof(float) * weightsSize_, true);

        d_dw_ = langHandle->allocDevMem((weightsSize_) * sizeof(float));
        
        Tracer::func_end("LinearLayer::LinearLayer");    
    }

    LinearLayer::~LinearLayer() {
        Tracer::func_begin("LinearLayer::~LinearLayer");
        free(h_weights_);
        langHandle_->freeDevPtr(d_weights_);
        langHandle_->freeDevPtr(d_dw_);
        Tracer::func_end("LinearLayer::~LinearLayer");    
    }

    void LinearLayer::doFw() {
        Tracer::func_begin("LinearLayer::doFw");
        
        assertBlasInvar(BlasRoutines::doMatMul(langHandle_, minibatchSize_, flattenedInputSize_, flattenedOutputSize_, 
                                                                                    d_input_, d_weights_, d_output_));
        Tracer::func_end("LinearLayer::doFw");    
    }

    void LinearLayer::doBw() {
        Tracer::func_begin("LinearLayer::doBw");
        
        assertBlasInvar(BlasRoutines::doMatMulTraB(langHandle_, minibatchSize_, flattenedOutputSize_, flattenedInputSize_, 
                                                                                    d_dy_, d_weights_, d_dx_));
        assertBlasInvar(BlasRoutines::doMatMulTraA(langHandle_, flattenedInputSize_, minibatchSize_, flattenedOutputSize_, d_input_, 
                                                                                    d_dy_, d_dw_));

        // update weights
        // w = w - learningRate * dw
        // OR w = - learningRate * dw + w
        float learningRate = -0.23;
        assertBlasInvar(BlasRoutines::doAxpy(langHandle_, 1 * weightsSize_, &learningRate, d_dw_, d_weights_));

        Tracer::func_end("LinearLayer::doBw");    
    }

    Mlp::Mlp(LangHandle *langHandle, Timer* timer, MlpIO mlpIO)
    : langHandle_(langHandle), timer_(timer), mlpIO_( std::move(mlpIO) ) { 
        
        Tracer::func_begin("Mlp::Mlp");
        
        LinearLayer *currentLinearLayer = 0;
        float *d_input = 0, *d_output = 0, *d_prevOutput = 0;
        float *d_dx, *d_dy, *d_prevDy;
        for(int i=0; i< mlpIO_.noOfLayers; i++) {
            int outputSize = mlpIO_.minibatchSize * mlpIO_.layerOutputCount[i];
            if(i == (mlpIO_.noOfLayers-1)) {
                d_output = mlpIO_.d_mlpOutput;
                d_dy = mlpIO_.d_mlpDy;                    
            } else {
                d_output = langHandle->allocDevMem((outputSize) * sizeof(float));
                d_dy = langHandle->allocDevMem((outputSize) * sizeof(float));
                outputAllocations_.push_back(d_output);
                outputAllocations_.push_back(d_dy);
            }

            if(i == 0) {
                d_input = mlpIO_.d_mlpInput;
                d_dx = mlpIO_.d_mlpDx;
                currentLinearLayer = new LinearLayer(langHandle, timer, mlpIO_.minibatchSize, 
                                            mlpIO_.flaInputSize, mlpIO_.layerOutputCount[i], d_input, d_output, d_dx, d_dy);
            } else {
                d_input = d_prevOutput;
                d_dx = d_prevDy;
                currentLinearLayer = new LinearLayer(langHandle, timer, mlpIO_.minibatchSize, 
                                            mlpIO_.layerOutputCount[i-1], mlpIO_.layerOutputCount[i], d_input, d_output, d_dx, d_dy);
            }          
            d_prevOutput = d_output;     
            d_prevDy = d_dy; 
            linearLayers_.push_back(currentLinearLayer);
        }
        Tracer::func_end("Mlp::Mlp");    
    }

    Mlp::~Mlp() {
        Tracer::func_begin("Mlp::~Mlp");
        int size = outputAllocations_.size();
        for(int i=0; i<size; i++) {
            langHandle_->freeDevPtr(outputAllocations_[i]);
        }
        Tracer::func_end("Mlp::~Mlp");    
    }

    void Mlp::doFw() {
        Tracer::func_begin("Mlp::doFw");

        for(int i=0; i< mlpIO_.noOfLayers; i++) {
            linearLayers_[i]->doFw();
        }
        Tracer::func_end("Mlp::doFw");    
    } 


    void Mlp::doBw() {
        Tracer::func_begin("Mlp::doBw");

        for(int i=0; i< mlpIO_.noOfLayers; i++) {
            linearLayers_[i]->doBw();
        }
        Tracer::func_end("Mlp::doBw");    
    }
};