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

#include "basic_mem_ops.h"

namespace dl_cifar::common {
    void BasicMemAllocs::createWeights(LangHandle *langHandle, float *&h_weights, float *&d_weights, float *&d_dw, std::vector<size_t> weightsTensorDims) {
        Tracer::func_begin("BasicMemAllocs::createWeights");

        size_t weightsSize    = std::accumulate(std::begin(weightsTensorDims), std::end(weightsTensorDims), 1, std::multiplies<size_t>());
        h_weights = (float*)calloc(weightsSize, sizeof(float));  
        ImageProcessor::initImage(h_weights, weightsSize);
        d_weights = langHandle->allocDevMem((weightsSize) * sizeof(float));
        d_dw      = langHandle->allocDevMem((weightsSize) * sizeof(float));

        langHandle->memCpyH2D(d_weights, h_weights, sizeof(float) * weightsSize, true);

        Tracer::func_end("BasicMemAllocs::createWeights");    
    }

    void BasicMemAllocs::createWeights(LangHandle *langHandle, float *&h_weights, float *&d_weights, float *&d_dw, int weightsTensorDims[]) {
        Tracer::func_begin("BasicMemAllocs::createWeights");

        int weightsSize = weightsTensorDims[0] * weightsTensorDims[1] * weightsTensorDims[2] * weightsTensorDims[3];
        h_weights = (float*)calloc(weightsSize, sizeof(float));  
        ImageProcessor::initImage(h_weights, weightsSize);
        d_weights = langHandle->allocDevMem((weightsSize) * sizeof(float));
        d_dw = langHandle->allocDevMem((weightsSize) * sizeof(float));

        langHandle->memCpyH2D(d_weights, h_weights, sizeof(float) * weightsSize, true);

        Tracer::func_end("BasicMemAllocs::createWeights");    
    }

    void BasicMemAllocs::destroyWeights(LangHandle *langHandle, float *h_weights, float *d_weights, float *d_dw) {
        Tracer::func_begin("BasicMemAllocs::destroyWeights");

        free(h_weights);
        langHandle->freeDevPtr(d_weights);
        langHandle->freeDevPtr(d_dw);


        Tracer::func_end("BasicMemAllocs::destroyWeights");    
    }

    void BasicMemAllocs::allocInputsOutputs(LangHandle *langHandle, float *&d_input, float *&d_output, float *&d_dx, float *&d_dy, int inputSize, int outputSize) {
        Tracer::func_begin("BasicMemAllocs::allocInputsOutputs");

        d_input  = langHandle->allocDevMem((inputSize)  * sizeof(float));
        d_output = langHandle->allocDevMem((outputSize) * sizeof(float));
        d_dx     = langHandle->allocDevMem((inputSize)  * sizeof(float));
        d_dy     = langHandle->allocDevMem((outputSize) * sizeof(float));

        Tracer::func_end("BasicMemAllocs::allocInputsOutputs");    
    }

    void BasicMemAllocs::allocAndRandInit(LangHandle *langHandle, float *&h_mem, float *&d_mem, float *&d_d_mem, int tensorDims[]) {
        Tracer::func_begin("BasicMemAllocs::allocAndRandInit");

        int size = tensorDims[0] * tensorDims[1] * tensorDims[2] * tensorDims[3];
        h_mem = (float*)calloc(size, sizeof(float));  
        ImageProcessor::initImage(h_mem, size);
        d_mem   = langHandle->allocDevMem((size) * sizeof(float));
        d_d_mem = langHandle->allocDevMem((size) * sizeof(float));

        langHandle->memCpyH2D(d_mem, h_mem, sizeof(float) * size, true);

        Tracer::func_end("BasicMemAllocs::allocAndRandInit");    
    }

    void BasicMemAllocs::setupImage(LangHandle *langHandle, float *&h_mem, float *&d_mem, int size) {
        Tracer::func_begin("BasicMemAllocs::setupImage");

        ImageProcessor::initImage(h_mem,   size);
        langHandle->memCpyH2D(d_mem, h_mem, sizeof(float) * size, true);

        Tracer::func_end("BasicMemAllocs::setupImage");    
    }
};