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
#include "timing.h"
#include "tracing.h"
#include "handle.h"
using namespace dl_cifar::common;


__global__ void execLNormKernelFw(const float * const d_input, float *d_output, int batchSize,
                                                            int noOfEmbs, int hiddenSize, float gamma, float beta);
__global__ void execLNormKernelBw(const float * const d_input, float *d_output, float *d_d_input, 
                const float * const d_d_output, int batchSize, int noOfEmbs, int embSize, float gamma, float beta);
                                                            

class LNormLayer {
    private:
    Timer* timer_;
        float gamma_, beta_;

        const float * const d_input_;
        const float * const d_d_output_;
        float *d_output_, *d_d_input_;
        int batchSize_, noOfEmbs_, embSize_;

        int blockSize = 128;
        int gridSize; 

    public:
        LNormLayer(LangHandle *langHandle, Timer* timer, const float * const d_input, float *d_output, float *d_d_input, const float * const d_d_output, 
                            int batchSize, int noOfEmbs, int embSize);
        void doFw();
        void doBw();
        ~LNormLayer();
};

class LNormLayerController { 
    public:
        static void execute();
};

#endif