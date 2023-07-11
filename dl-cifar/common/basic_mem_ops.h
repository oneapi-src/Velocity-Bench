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

#ifndef DL_CIFAR_BASIC_MEM_OPS_H_
#define DL_CIFAR_BASIC_MEM_OPS_H_

#include <vector>
#include <numeric>
#include "timing.h"
#include "tracing.h"
#include "handle.h"
#include "image_processing.h"


namespace dl_cifar::common {
    class BasicMemAllocs {
        public:
            static void createWeights(LangHandle *langHandle, float *&h_weights, float *&d_weights, float *&d_dw, std::vector<size_t> weightsTensorDims);
            static void createWeights(LangHandle *langHandle, float *&h_weights, float *&d_weights, float *&d_dw, int weightsTensorDims[]) ;
            static void destroyWeights(LangHandle *langHandle, float *h_weights, float *d_weights, float *d_dw);

            static void allocInputsOutputs(LangHandle *langHandle, float *&d_input, float *&d_output, float *&d_dx, float *&d_dy, int inputSize, int outputSize);
            static void allocAndRandInit(LangHandle *langHandle, float *&h_mem, float *&d_mem, float *&d_d_mem, int tensorDims[]);

            static void setupImage(LangHandle *langHandle, float *&h_mem, float *&d_mem, int size);

    };
};


#endif