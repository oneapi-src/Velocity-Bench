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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../error_handling.h"
#include "lnorm_layer.h"



__global__ void execLNormKernelFw(const float * const d_input, float *d_output, int batchSize, int noOfEmbs, int embSize, float gamma, float beta)
{
    // Get our global thread ID
    //int id = blockIdx.x*blockDim.x+threadIdx.x;

    int offset = (blockIdx.x*blockDim.x+threadIdx.x) * embSize;
    //float * latentVecPtr = d_input + offset;
 
    // Make sure we do not go out of bounds
    // if (id < n)
    //     c[id] = a[id] + b[id];

    float epsilon = 0.0000001;
    if(offset < batchSize * noOfEmbs * embSize) {
        float mean = 0;
        for(int i=0; i<embSize; i++) {
            mean += d_input[offset+i];
        }
        mean /= embSize;

        float variance = 0;
        for(int i=0; i<embSize; i++) {
            variance += (d_input[offset+i] - mean) * (d_input[offset+i] - mean);
        }
        variance /= embSize;

        for(int i=0; i<embSize; i++) {
            d_output[offset+i] = gamma * ((d_input[offset+i] - mean) / std::sqrt(variance + epsilon)) + beta;
        }
    }
}


__global__ void execLNormKernelBw(const float * const d_input, float *d_output, float *d_d_input, const float * const d_d_output, 
        int batchSize, int noOfEmbs, int embSize, float gamma, float beta)
{
    int offset = (blockIdx.x*blockDim.x+threadIdx.x) * embSize;
    float epsilon = 0.0000001;

    if(offset < batchSize * noOfEmbs * embSize) {
        float mean = 0;
        for(int i=0; i<embSize; i++) {
            mean += d_input[offset+i];
        }
        mean /= embSize;

        float variance = 0;
        for(int i=0; i<embSize; i++) {
            variance += (d_input[offset+i] - mean) * (d_input[offset+i] - mean);
        }
        variance /= embSize;





        float dBeta = 0;
        for(int i=0; i<embSize; i++) {
            dBeta += d_d_output[offset+i];
        }

        float dGamma = 0;
        for(int i=0; i<embSize; i++) {
            dGamma += d_d_output[offset+i] * ((d_input[offset+i] - mean) / std::sqrt(variance + epsilon));
        }

        float dVariance = 0;
        for(int i=0; i<embSize; i++) {
            dVariance += (d_d_output[offset+i] * gamma * (d_input[offset+i] - mean)) * (-1/embSize) * std::pow((variance + epsilon), (-3/2));
        }

        float dMeanFirstTerm = 0;
        for(int i=0; i<embSize; i++) {
            dMeanFirstTerm += (d_d_output[offset+i] * gamma * ((-1)/(std::sqrt(variance + epsilon)))); 
        }
        float dMeanSecondTerm = 0;
        for(int i=0; i<embSize; i++) {
            dMeanSecondTerm += dVariance * (-2) * (d_input[offset+i] - mean); 
        }
        dMeanSecondTerm = (dVariance * dMeanSecondTerm)/2;
        float dMean = dMeanFirstTerm + dMeanSecondTerm;

        for(int i=0; i<embSize; i++) {
            d_d_input[offset+i] = d_d_output[offset+i] * gamma * 1/(std::sqrt(variance + epsilon)) + 
                                    dVariance * 2 * (d_input[offset+i] - mean)/embSize +
                                    dMean/embSize; 
        }


    }
} 


LNormLayer::LNormLayer(LangHandle *langHandle, Timer* timer, const float * const d_input, float *d_output, float *d_d_input, const float * const d_d_output, int batchSize,
                                                    int noOfEmbs, int embSize)
    : timer_(timer), d_input_(d_input), d_output_(d_output), d_d_input_(d_d_input), d_d_output_(d_d_output), batchSize_(batchSize), noOfEmbs_(noOfEmbs), embSize_(embSize) {

        Tracer::func_begin("LNormLayer::LNormLayer");
    srand( (unsigned)time( NULL ) );
    gamma_ = 1.0;
    beta_  = 0.0;
    //std::cout << "gamma = " << alpha << std::endl;
    //std::cout << "beta = " << beta << std::endl;

    gridSize = (int)ceil((float)noOfEmbs/blockSize);

    Tracer::func_end("LNormLayer::LNormLayer");   
}

void LNormLayer::doFw() {
    Tracer::func_begin("LNormLayer::doFw");
    execLNormKernelFw<<<gridSize, blockSize>>>(d_input_, d_output_, batchSize_, noOfEmbs_, embSize_, gamma_, beta_);
    assertDevApiInvar(hipDeviceSynchronize());
    assertDevApiInvar(hipGetLastError());
    Tracer::func_end("LNormLayer::doFw");   
}

void LNormLayer::doBw() {
    Tracer::func_begin("LNormLayer::doBw");
    execLNormKernelBw<<<gridSize, blockSize>>>(d_input_, d_output_, d_d_input_, d_d_output_, batchSize_, noOfEmbs_, embSize_, gamma_, beta_);
    assertDevApiInvar(hipDeviceSynchronize());
    assertDevApiInvar(hipGetLastError());
    Tracer::func_end("LNormLayer::doBw");   
}

LNormLayer::~LNormLayer() {
    Tracer::func_begin("LNormLayer::~LNormLayer");

    Tracer::func_end("LNormLayer::~LNormLayer");
}

void LNormLayerController::execute() {
    Timer* timer = new Timer();

    LangHandle *langHandle = new LangHandle(timer);

    int vitL16_imgWidth = 384;
    int vitL16_imgHeight = 384;
    int patchSize = 16;
    int batchSize = 512;
    int noOfEmbs = (vitL16_imgWidth*vitL16_imgHeight)/(patchSize*patchSize);    //should be 576
    int vitL16_noOfPatches_fullBatch = batchSize * noOfEmbs;    //should be 576

    int embSize = 512;
    int inputSize   = vitL16_noOfPatches_fullBatch * embSize;
    float *h_input   = (float*)calloc(inputSize,   sizeof(float));  
    float *d_input, *d_d_input;
    assertDevApiInvar(hipMalloc((void**)&(d_input),   (inputSize) * sizeof(float)));
    assertDevApiInvar(hipMalloc((void**)&(d_d_input), (inputSize) * sizeof(float)));
    assertDevApiInvar(hipMemcpy(d_input,   h_input,   sizeof(float) * inputSize,   hipMemcpyHostToDevice));
    assertDevApiInvar(hipDeviceSynchronize());

    int outputSize   = inputSize;
    float *h_d_output   = (float*)calloc(outputSize,   sizeof(float));  
    float *d_output, *d_d_output;
    assertDevApiInvar(hipMalloc((void**)&(d_output),   (outputSize) * sizeof(float)));
    assertDevApiInvar(hipMalloc((void**)&(d_d_output), (outputSize) * sizeof(float)));

    LNormLayer *lNormLayer = new LNormLayer(langHandle, timer, d_input, d_output, d_d_input, d_d_output, batchSize, noOfEmbs, embSize);

    int iterCount = 3;

    for(int i=0; i<iterCount; i++) {
        // for some reason the compiler is not liking calls to ImageProcessor::initImage() from here
        //ImageProcessor::initImage(h_input, inputSize);
        assertDevApiInvar(hipMemcpy(d_input, h_input, sizeof(float) * inputSize, hipMemcpyHostToDevice));
        assertDevApiInvar(hipDeviceSynchronize());
        lNormLayer->doFw();

        // for some reason the compiler is not liking calls to ImageProcessor::initImage() from here
        //ImageProcessor::initImage(h_d_output, outputSize);
        assertDevApiInvar(hipMemcpy(d_d_output, h_d_output, sizeof(float) * outputSize, hipMemcpyHostToDevice));
        assertDevApiInvar(hipDeviceSynchronize());

        lNormLayer->doBw();
    }

    delete lNormLayer;
    free(h_d_output);
    free(h_input);
}











