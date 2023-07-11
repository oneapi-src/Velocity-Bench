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

#ifndef DL_CIFAR_MLP_H_
#define DL_CIFAR_MLP_H_

#include <vector>
#include <iostream>
#include "blas_routines.h"
#include "tracing.h"
#include "timing.h"
#include "handle.h"
#include "error_handling.h"
#include "image_processing.h"

namespace dl_cifar::common {

    class LinearLayer {
        private:
            Timer* timer_;
            LangHandle *langHandle_;
            int minibatchSize_, flattenedInputSize_, flattenedOutputSize_;
            int inputSize_, weightsSize_, outputSize_;

            float *d_input_, *h_weights_, *d_weights_, *d_output_;
            float *d_dx_, *d_dw_, *d_dy_;

        public:
            LinearLayer(LangHandle *langHandle, Timer* timer, int minibatchSize, int flattenedInputSize, int flattenedOutputSize, 
                                                            float *d_input, float *d_output, float *d_dx, float *d_dy);

            ~LinearLayer();

            void doFw();
            void doBw();
    };

    struct MlpIO {
        int    minibatchSize;
        int    flaInputSize;
        int    flaOutputSize;
        float *d_mlpInput;
        float *d_mlpOutput;
        float *d_mlpDx;
        float *d_mlpDy;

        int noOfLayers;
        std::vector<int> layerOutputCount;
    };

    class Mlp {
        private:
            Timer* timer_;
            LangHandle *langHandle_;
            MlpIO mlpIO_;        
            std::vector<LinearLayer *> linearLayers_;
            std::vector<float *> outputAllocations_;

        public:
            Mlp(LangHandle *langHandle, Timer* timer, MlpIO mlpIO);
            ~Mlp();

            void doFw();
            void doBw();
    };


    class MlpController {   

        public:
            static void execute() {
                Timer* timer = new Timer();

                LangHandle *langHandle = new LangHandle(timer);
                
                int N = 256;          // minibatch size
                int D = 512;      // flattened input size
                int M = 512;      // flattened output size

                int inputSize   = N*D;
                int weightsSize = D*M;
                int outputSize  = N*M;

            
                int noOfLayers = 8;
                std::vector<int> layerOutputCounts;
                for(int i=0; i<noOfLayers; i++) {
                    layerOutputCounts.push_back(512);
                }

                float *h_input   = (float*)calloc(inputSize,   sizeof(float));  
                float *d_input = langHandle->allocDevMem((inputSize) * sizeof(float));
                float *d_output = langHandle->allocDevMem((outputSize) * sizeof(float));

                float *h_dy   = (float*)calloc(outputSize,   sizeof(float));  
                float *d_dx = langHandle->allocDevMem((inputSize) * sizeof(float));
                float *d_dy = langHandle->allocDevMem((outputSize) * sizeof(float));
                

                MlpIO mlpIO; 
                mlpIO.minibatchSize = N;
                mlpIO.flaInputSize = D;
                mlpIO.flaOutputSize = M;
                mlpIO.d_mlpInput = d_input;
                mlpIO.d_mlpOutput = d_output;
                mlpIO.d_mlpDx = d_dx;
                mlpIO.d_mlpDy = d_dy;
                mlpIO.noOfLayers = noOfLayers;
                mlpIO.layerOutputCount = layerOutputCounts;


                Mlp *mlp = new Mlp(langHandle, timer, mlpIO);

                //---------- train-----------------------------------------

                for(int i=0; i<5; i++) {
                    ImageProcessor::initImage(h_input,   inputSize);
                    langHandle->memCpyH2D(d_input, h_input, sizeof(float) * inputSize, true);
                    mlp->doFw();

                    ImageProcessor::initImage(h_dy,   outputSize);
                    langHandle->memCpyH2D(d_dy, h_dy, sizeof(float) * outputSize, true);
                    mlp->doBw();
                }
            }
    };
};


#endif