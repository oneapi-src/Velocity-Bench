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

#include "lnorm_layer.h"




LNormLayer::LNormLayer(LangHandle *langHandle, Timer* timer, const float * const d_input, float *d_output, 
                                        float *d_d_input, const float * const d_d_output, int batchSize, int noOfEmbs, int embSize)
    : langHandle_(langHandle), timer_(timer), d_input_(d_input), d_output_(d_output), d_d_input_(d_d_input), 
    d_d_output_(d_d_output), batchSize_(batchSize), noOfEmbs_(noOfEmbs), embSize_(embSize) {
    
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

    int embSize = embSize_;
    int batchSize = batchSize_; 
    int noOfEmbs  = noOfEmbs_;
    float gamma = gamma_;
    float beta = beta_;

    const float * const d_input = d_input_;
    float *d_output = d_output_;

    langHandle_->getSyclQueue()->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range{static_cast<size_t>(gridSize), static_cast<size_t>(blockSize)},  [=](sycl::id<2> idx) {

            int offset = (idx[0]*128+idx[1]) * embSize;

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
        });
    }).wait();


    Tracer::func_end("LNormLayer::doFw");   
}

void LNormLayer::doBw() {
    Tracer::func_begin("LNormLayer::doBw");
    
    int embSize = embSize_;
    int batchSize = batchSize_; 
    int noOfEmbs  = noOfEmbs_;
    float gamma = gamma_;
    float beta = beta_;

    const float * const d_input = d_input_;
    const float * const d_d_output = d_d_output_;
    float *d_d_input = d_d_input_;

    langHandle_->getSyclQueue()->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range{static_cast<size_t>(gridSize), static_cast<size_t>(blockSize)},  [=](sycl::id<2> idx) {

            int offset = (idx[0]*128+idx[1]) * embSize;

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
                    dVariance += (d_d_output[offset+i] * gamma * (d_input[offset+i] - mean)) * (-1/(float)embSize) * std::pow((variance + epsilon), (-3/2));
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
        });
    }).wait();    

    Tracer::func_end("LNormLayer::doBw");   
}

LNormLayer::~LNormLayer() {
    Tracer::func_begin("LNormLayer::~LNormLayer");

    Tracer::func_begin("LNormLayer::~LNormLayer");
}












