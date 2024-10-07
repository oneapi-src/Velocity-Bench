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

#ifndef DL_CIFAR_LNORM_LAYER_H_
#define DL_CIFAR_LNORM_LAYER_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sycl/sycl.hpp>

#include "tracing.h"
#include "timing.h"
#include "handle.h"

using namespace dl_cifar::common;

// __global__ void execLNormKernelFw(const float * const d_input, float *d_output, int batchSize,
//                                                             int noOfEmbs, int hiddenSize, float gamma, float beta);
// __global__ void execLNormKernelBw(const float * const d_input, float *d_output, float *d_d_input, 
//                 const float * const d_d_output, int batchSize, int noOfEmbs, int embSize, float gamma, float beta);
                                                            

class LNormLayer {
    private:
        LangHandle *langHandle_;
        Timer* timer_;

        float gamma_, beta_;

        const float * const d_input_;
        const float * const d_d_output_;
        float *d_output_, *d_d_input_;
        int batchSize_, noOfEmbs_, embSize_;

        int blockSize = 128;
        int gridSize; 

    public:
        LNormLayer(LangHandle *langHandle, Timer* timer, const float * const d_input, float *d_output, 
                                    float *d_d_input, const float * const d_d_output, int batchSize, int noOfEmbs, int embSize);                    
        void doFw();
        void doBw();
        ~LNormLayer();
};

class LNormLayerController { 
    public:
        static void execute() {
            Timer* timer = new Timer();

            LangHandle *langHandle = new LangHandle(timer);

            sycl::device* dht = new sycl::device(sycl::gpu_selector_v);
            sycl::context context(*dht);
            sycl::queue sycl_queue(context, *dht);

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
            d_input   = (float *)sycl::malloc_device(inputSize*sizeof(float),   sycl_queue);
            d_d_input   = (float *)sycl::malloc_device(inputSize*sizeof(float),   sycl_queue);
            sycl_queue.memcpy(d_input, h_input, sizeof(float) * inputSize).wait();

            int outputSize   = inputSize;
            float *h_d_output   = (float*)calloc(outputSize,   sizeof(float));  
            float *d_output, *d_d_output;
            d_output   = (float *)sycl::malloc_device(outputSize*sizeof(float),   sycl_queue);
            d_d_output   = (float *)sycl::malloc_device(outputSize*sizeof(float),   sycl_queue);

            LNormLayer *lNormLayer = new LNormLayer(langHandle, timer, d_input, d_output, d_d_input, d_d_output, batchSize, 
                                                        noOfEmbs, embSize);

            int iterCount = 3;

            for(int i=0; i<iterCount; i++) {
                // for some reason the compiler is not liking calls to ImageProcessor::initImage() from here
                //ImageProcessor::initImage(h_input, inputSize);
                sycl_queue.memcpy(d_input, h_input, sizeof(float) * inputSize).wait();
                lNormLayer->doFw();

                // for some reason the compiler is not liking calls to ImageProcessor::initImage() from here
                //ImageProcessor::initImage(h_d_output, outputSize);
                sycl_queue.memcpy(d_d_output, h_d_output, sizeof(float) * outputSize).wait();
                lNormLayer->doBw();
            }

            delete lNormLayer;
            free(h_d_output);
            free(h_input);

        }

};

#endif