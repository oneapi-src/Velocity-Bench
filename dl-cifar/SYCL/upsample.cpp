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
#include <cmath>
#include "error_handling.h"
#include "upsample.h"



void Upsampler::upsample(LangHandle *langHandle, float *d_src, float *d_dst, int noOfImgs, 
                            int noOfChannels, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    Tracer::func_begin("Upsampler::upsample");

    const int segmentLen = 8;
    int totalNoOfDstPixels = noOfImgs * noOfChannels * dstWidth * dstHeight;
    int totalNoOfSegments = totalNoOfDstPixels/segmentLen;
    const int blockSize = 128; 
    int gridSize = (int)ceil((float)totalNoOfSegments/blockSize);
    
    // std::cout<< "noOfImgs = " << noOfImgs << std::endl;
    // std::cout<< "blockSize = " << blockSize << std::endl;
    // std::cout<< "gridSize = " << gridSize << std::endl;


    langHandle->getSyclQueue()->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range{static_cast<size_t>(gridSize), static_cast<size_t>(blockSize)},  [=](sycl::id<2> idx) {

            int linearSegmentIdx = idx[0]*blockSize+idx[1];


            int src_noOfPixelsPerImg = srcWidth * srcHeight;
            int dst_noOfPixelsPerImg = dstWidth * dstHeight;

            int dst_pixCntUntilSegment = linearSegmentIdx * segmentLen;    
            int dst_noOfSegmentsPerImg = (dstWidth * dstWidth)/segmentLen;

            int dst_imgIdx = dst_pixCntUntilSegment/dst_noOfPixelsPerImg;
            int src_imgIdx = dst_imgIdx;

            int dst_relPixIdxOfSegment = dst_pixCntUntilSegment - (dst_imgIdx*dst_noOfPixelsPerImg);
            int dst_noOfSegmentsPerWidth = dstWidth/segmentLen;
            int dst_noOfSegmentsPerHeight = dstHeight/segmentLen;

            int dst_relPixY_Segment = dst_relPixIdxOfSegment/dstWidth;
            int dst_relPixX_Segment = dst_relPixIdxOfSegment%dstWidth;


            for(int relPixIdxInSegment=0; relPixIdxInSegment<segmentLen; relPixIdxInSegment++) {

                int pixelOX = dst_relPixX_Segment + relPixIdxInSegment;
                int pixelOY = dst_relPixY_Segment + relPixIdxInSegment;

                int pixelIX = static_cast<int>(std::round((pixelOX * srcWidth)/((float)dstWidth)));
                int pixelIY = static_cast<int>(std::round((pixelOY * srcWidth)/((float)dstWidth)));

                d_dst[dst_pixCntUntilSegment + relPixIdxInSegment] = d_src[src_imgIdx*src_noOfPixelsPerImg + (pixelIY*srcWidth) + pixelIX];
            }
        });
    }).wait();

    Tracer::func_end("Upsampler::upsample");   
}

void Upsampler::hostUpsample(int inputRes, int outputRes, float *input, float* output) {
    Tracer::func_begin("Upsampler::hostUpsample");         

    int scale = outputRes/inputRes;

    typedef float MatrixRow[2]; 
    MatrixRow* mInput = (MatrixRow*) input;
    MatrixRow* mOutput = (MatrixRow*) output;
    
    for(int posOX=0; posOX<scale*outputRes; posOX++) {
        for(int posOY=0; posOY<scale*outputRes; posOY++) {

            float posIX = (posOX * inputRes)/((float)outputRes);
            float posIY = (posOY * inputRes)/((float)outputRes);

            float floorIX = std::floor(posIX);
            float ceilIX  = std::ceil(posIX);

            float floorIY = std::floor(posIY);
            float ceilIY  = std::ceil(posIY);

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
    Tracer::func_end("Upsampler::hostUpsample");    
}


Upsampler::~Upsampler() {
    Tracer::func_begin("Upsampler::~Upsampler");

    Tracer::func_end("Upsampler::~Upsampler");   
}



void UpsamplerController::execute() {
    Timer* timer = new Timer();            
    LangHandle *langHandle = new LangHandle(timer);

    sycl::device* dht = new sycl::device(sycl::gpu_selector_v);
    sycl::context context(*dht);
    sycl::queue sycl_queue(context, *dht);

    int noOfImgs = 768; //512; //128;
    int noOfChannels = 3;
    int srcWidth = 32;
    int srcHeight = 32;
    int dstWidth = 384;
    int dstHeight = 384;

    int inputSize = noOfImgs * noOfChannels * srcWidth * srcHeight;
    int outputSize = noOfImgs * noOfChannels * dstWidth * dstHeight;

    float *d_src, *d_dst;
    d_src   = (float *)sycl::malloc_device(inputSize*sizeof(float), sycl_queue);
    d_dst   = (float *)sycl::malloc_device(outputSize*sizeof(float), sycl_queue);

    Upsampler::upsample(langHandle, d_src, d_dst, noOfImgs, noOfChannels, srcWidth, srcHeight, dstWidth, dstHeight);

    sycl::free(d_src, sycl_queue);   
    sycl::free(d_dst, sycl_queue);   
    delete langHandle;

}











