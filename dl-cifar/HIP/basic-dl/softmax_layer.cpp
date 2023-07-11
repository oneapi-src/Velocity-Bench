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

#include "softmax_layer.h"

SoftmaxLayer::SoftmaxLayer(LangHandle *langHandle, Timer* timer,
     int softmaxInputTensorDims[4], int softmaxOutputTensorDims[4],
     float *d_input, float *d_d_input, float *d_output, float *d_d_output)
     : langHandle_(langHandle), timer_(timer), d_input_(d_input), d_d_input_(d_d_input), 
       d_output_(d_output), d_d_output_(d_d_output) {

    softmaxInputTensorDims_ = softmaxInputTensorDims;
    softmaxOutputTensorDims_ = softmaxOutputTensorDims;

    softmaxInputStrideDims_[3] = 1;
    softmaxInputStrideDims_[2] = softmaxInputTensorDims_[3];
    softmaxInputStrideDims_[1] = softmaxInputTensorDims_[2] * softmaxInputTensorDims_[3];
    softmaxInputStrideDims_[0] = softmaxInputTensorDims_[1] * softmaxInputTensorDims_[2] * softmaxInputTensorDims_[3];

    softmaxOutputStrideDims_[3] = 1;
    softmaxOutputStrideDims_[2] = softmaxOutputTensorDims_[3];
    softmaxOutputStrideDims_[1] = softmaxOutputTensorDims_[2] * softmaxOutputTensorDims_[3];
    softmaxOutputStrideDims_[0] = softmaxOutputTensorDims_[1] * softmaxOutputTensorDims_[2] * softmaxOutputTensorDims_[3];

    assertDlApiInvar(miopenCreateTensorDescriptor(&softMaxInputDesc_));
    assertDlApiInvar(miopenSetTensorDescriptor(softMaxInputDesc_, miopenFloat, 4, 
                                                    softmaxInputTensorDims_, softmaxInputStrideDims_));

    assertDlApiInvar(miopenCreateTensorDescriptor(&softMaxOutputDesc_));
    assertDlApiInvar(miopenSetTensorDescriptor(softMaxOutputDesc_, miopenFloat, 4, 
                                                    softmaxOutputTensorDims_, softmaxOutputStrideDims_));
}

void SoftmaxLayer::doFw() {    
    assertDlApiInvar(miopenSoftmaxForward(*(langHandle_->getMiopenHandle()),
                                    (void*)(&alpha),
                                    softMaxInputDesc_,
                                    d_input_,
                                    (void*)(&beta),
                                    softMaxOutputDesc_,
                                    d_output_));
    assertDlApiInvar(hipDeviceSynchronize());
}

void SoftmaxLayer::doBw() {
    assertDlApiInvar(miopenSoftmaxBackward(*(langHandle_->getMiopenHandle()),
                                    (void*)(&alpha),
                                    softMaxOutputDesc_,
                                    d_output_,
                                    softMaxOutputDesc_,
                                    d_d_output_,
                                    (void*)(&beta),
                                    softMaxInputDesc_,
                                    d_d_input_));
    assertDlApiInvar(hipDeviceSynchronize());
}

SoftmaxLayer::~SoftmaxLayer() {

}