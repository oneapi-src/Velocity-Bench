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

#ifndef DL_CIFAR_IMAGE_PATCHING_H_
#define DL_CIFAR_IMAGE_PATCHING_H_

#include <iostream>
#include <cassert>
#include <vector>
#include <exception>
#include "tracing.h"
#include"handle.h"
#include "upsample.h"

namespace dl_cifar::common {
    class ImageProcessor {
        public:
            static void convertDevImgsToPatches(LangHandle *langHandle, float *d_inputImgs, float *d_imgPatches, int noOfImgs, int noOfChannels, int imgWidth, 
                                                                int imgHeight, int patchSize) {

                convertImgsToPatches(langHandle, d_inputImgs, d_imgPatches, noOfImgs, noOfChannels, imgWidth, 
                                                                imgHeight, patchSize, D2D);
            }

            static void convertHostImgsToPatches(LangHandle *langHandle, float *h_imputImgs, float *d_imgPatches, int noOfImgs, int noOfChannels, int imgWidth, 
                                                                int imgHeight, int patchSize) {

                convertImgsToPatches(langHandle, h_imputImgs, d_imgPatches, noOfImgs, noOfChannels, imgWidth, 
                                                                imgHeight, patchSize, H2D);
            }

            

            static void softSplitDevImagesToPatches(LangHandle *langHandle, float *h_imputImgs, float *d_imgPatches, int noOfImgs, int noOfChannels, int imgWidth, 
                                                                int imgHeight, int patchSize, int padding, int stride) {

                softSplitImagesToPatches(langHandle, h_imputImgs, d_imgPatches, noOfImgs, noOfChannels, imgWidth, 
                                                                imgHeight, patchSize, padding, stride, D2D);
            }

            static void softSplitHostImagesToPatches(LangHandle *langHandle, float *h_imputImgs, float *d_imgPatches, int noOfImgs, int noOfChannels, int imgWidth, 
                                                                int imgHeight, int patchSize, int padding, int stride) {

                softSplitImagesToPatches(langHandle, h_imputImgs, d_imgPatches, noOfImgs, noOfChannels, imgWidth, 
                                                                imgHeight, patchSize, padding, stride, H2D);
            }

            static void resize(LangHandle *langHandle, float *d_src, float *d_dst, int noOfImgs, 
                            int noOfChannels, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {

                Tracer::func_begin("ImageProcessor::resize");            

                Upsampler::upsample(langHandle, d_src, d_dst, noOfImgs, noOfChannels, srcWidth, srcHeight, dstWidth, dstHeight);

                // int dstSize = noOfImgs * noOfChannels * dstWidth * dstHeight;
                // float *h_dst   = (float*)calloc(dstSize,   sizeof(float));  
                // initImage(h_dst, dstSize);
                // langHandle->memCpyH2D(d_dst, h_dst, sizeof(float) * dstSize, true);

                // free(h_dst);
                Tracer::func_end("ImageProcessor::resize");    
            }

            static void resizeInHost(LangHandle *langHandle, float *h_src, float *h_dst, int noOfImgs, 
                            int noOfChannels, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
                                
                Tracer::func_begin("ImageProcessor::resizeinHost");            

                for(int imgIdx=0; imgIdx<noOfImgs; imgIdx++) {
                    for(int channelIdx=0; channelIdx<noOfChannels; channelIdx++) {
                        hostUpsample(srcWidth, dstWidth, 
                            h_src + (imgIdx*noOfChannels*srcWidth*srcHeight) + (channelIdx*srcWidth*srcHeight), 
                            h_dst + (imgIdx*noOfChannels*dstWidth*dstHeight) + (channelIdx*dstWidth*dstHeight));
                    }
                }
 
                Tracer::func_end("ImageProcessor::resizeinHost");    
            }

            

            static void initImage(float* image, int imageSize) {
                Tracer::func_begin("ImageProcessor::initImage");

                unsigned seed = 123456789;
                for (int index = 0; index < imageSize; index++) {
                    seed         = (1103515245 * seed + 12345) & 0xffffffff;
                    image[index] = float(seed) * 2.3283064e-10;  // 2^-32
                }
                Tracer::func_end("ImageProcessor::initImage");

            }


        private:
            static void hostUpsample(int inputRes, int outputRes, float *input, float* output) {
                Tracer::func_begin("ImageProcessor::upSample");         

                int scale = outputRes/inputRes;

                typedef float MatrixRow[2]; 
                MatrixRow* mInput = (MatrixRow*) input;
                MatrixRow* mOutput = (MatrixRow*) output;
                
                for(int posOX=0; posOX<scale*outputRes; posOX++) {
                    for(int posOY=0; posOY<scale*outputRes; posOY++) {

                        float posIX = (posOX * inputRes)/((float)outputRes);
                        float posIY = (posOY * inputRes)/((float)outputRes);

                        float floorIX = std::floor(posIX);
                        float ceilIX = std::ceil(posIX);

                        float floorIY = std::floor(posIY);
                        float ceilIY = std::ceil(posIY);

                        if((posIX-floorIX) < (ceilIX-posIX)) {
                            posIX = floorIX;
                        } else {
                            posIX = ceilIX;
                        }

                        if((posIY-floorIY) < (ceilIY-posIY)) {
                            posIY = floorIY;
                        } else {
                            posIY = ceilIY;
                        }

                        mOutput[posOX][posOY] = mInput[static_cast<int>(posIX)][static_cast<int>(posIY)];

                    }
                }
                Tracer::func_end("ImageProcessor::upSample");    
            }


            static void convertImgsToPatches(LangHandle *langHandle, float *inputImgs, float *d_imgPatches, int noOfImgs, int noOfChannels, 
                                                    int imgWidth, int imgHeight, int patchSize, MemcpyType memcpyType) {

                Tracer::func_begin("ImageProcessor::convertImgsToPatches");            

                int imgSize = noOfChannels * imgWidth * imgHeight;
                int noOfPatches = (imgWidth * imgHeight)/(patchSize * patchSize);

                for(int imgIdx=0; imgIdx<noOfImgs; imgIdx++) {
                    for(int patchIdx=0; patchIdx<noOfPatches; patchIdx++) {
                        for(int channelIdx=0; channelIdx<noOfChannels; channelIdx++) {
                            float *d_dst = d_imgPatches + (imgIdx * imgSize) 
                                + (patchIdx*patchSize*patchSize*noOfChannels) + (channelIdx*patchSize*patchSize);
                            float *hORd_src = inputImgs + (imgIdx * imgSize) + (channelIdx  *imgWidth * imgHeight) 
                                + (patchIdx*patchSize*patchSize);
                            langHandle->memCpy(d_dst, hORd_src, sizeof(float) * patchSize*patchSize, false, memcpyType);    
                        }
                    }
                }
                langHandle->synchronize();
                Tracer::func_end("ImageProcessor::convertImgsToPatches");    
            }


            static void softSplitImagesToPatches(LangHandle *langHandle, float *inputImgs, float *d_imgPatches, int noOfImgs, int noOfChannels, int imgWidth, 
                                                                int imgHeight, int patchSize, int padding, int stride, MemcpyType memcpyType) {

                Tracer::func_begin("ImageProcessor::softSplitImagesToPatches");            
                
                int imgSize       = noOfChannels * imgWidth * imgHeight;
                int paddedWidth   = imgWidth  + 2 * padding; 
                int paddedHeight  = imgHeight + 2 * padding;
                int paddedImgSize = noOfChannels * paddedWidth * paddedHeight;
                int noOfPatches   = (paddedWidth * paddedHeight) / (patchSize * patchSize);

                // we expect the appropriate padding has been passed to make noOfPatches whole
                assert((paddedWidth * paddedHeight) % (patchSize * patchSize) == 0);

                int runLength = patchSize;
                for(int imgIdx=0; imgIdx<noOfImgs; imgIdx++) {
                    for(int patchIdx=0; patchIdx<noOfPatches; patchIdx++) {
                        for(int channelIdx=0; channelIdx<noOfChannels; channelIdx++) {
                            float *d_dstPatch = d_imgPatches  + (imgIdx * paddedImgSize)
                                + (patchIdx*patchSize*patchSize*noOfChannels) 
                                + (channelIdx*patchSize*patchSize);

                            float *hORd_src = inputImgs + (imgIdx * imgSize) + (channelIdx  *imgWidth * imgHeight)
                                + (patchIdx*patchSize*patchSize);

                            int noOfRuns = patchSize;
                            for(int runIdx=0; runIdx<noOfRuns; runIdx++) {
                                

                            }
                        }
                    }
                }
                langHandle->synchronize();
                Tracer::func_end("ImageProcessor::softSplitImagesToPatches");    
            }



    };

    class ImagePatcherController {

    };
};
#endif