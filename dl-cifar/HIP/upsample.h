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

#ifndef DL_CIFAR_UPSAMPLE_H_
#define DL_CIFAR_UPSAMPLE_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tracing.h"
#include "timing.h"
#include "handle.h"

using namespace dl_cifar::common;


__global__ void upsampleKernel3(float *d_src, float *d_dst, int noOfImgs, 
                            int noOfChannels, int srcWidth, int srcHeight, int dstWidth, int dstHeight, int segmentLen);                                                     
                                 

class Upsampler {
     public:
        static void upsample(LangHandle *langHandle, float *d_src, float *d_dst, int noOfImgs, 
                            int noOfChannels, int srcWidth, int srcHeight, int dstWidth, int dstHeight);

        static void hostUpsample(int inputRes, int outputRes, float *input, float* output);

        ~Upsampler();
};

class UpsamplerController { 
    public:
        static void execute();
};

#endif