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

#ifndef DL_CIFAR_CIFAR_READER_H_
#define DL_CIFAR_CIFAR_READER_H_

#include <iostream>
#include <fstream>
#include <exception>

namespace dl_cifar::common {
    class CifarReader {
        public:
            static void readCifarFile(std::string full_path, int noOfImages, float *h_dst) {
                
                int labelSize            = 1;
                int cifarImgWidth        = 32;
                int cifarImgHeight       = 32;
                int cifarImgNoOfChannels = 3;
                int imageSize          = cifarImgNoOfChannels * cifarImgWidth * cifarImgHeight;    
                int labelPlusImageSize = labelSize + imageSize;
                int cifarImgDataSize   = noOfImages * imageSize;
                int readSize           = noOfImages * (labelSize + imageSize);   

                typedef unsigned char uchar;
                uchar* charIm = new uchar[readSize];
                //float* floatIm = new float[reqSize-labelSize];

                std::ifstream file(full_path, std::ios::binary);

                if(file.is_open()) {
                    file.read((char *)charIm, readSize);

                    for(int imgCounter=0; imgCounter<noOfImages; imgCounter++) {
                        for(int imgDataCounter=0; imgDataCounter<imageSize; imgDataCounter++) {   
                            h_dst[imgCounter*imageSize + imgDataCounter] = 
                                        static_cast<float>(charIm[imgCounter*labelPlusImageSize + labelSize + imgDataCounter]);
                        }    
                    }                
                    delete[] charIm;              
                } else {
                    throw std::runtime_error("Unable to open file `" + full_path + "`!");
                }
            }
    };

    class CifarReaderController {
        public:
            static void execute() {
                std::string cifar_dataset_dir = "../../datasets/cifar-10-batches-bin/";
                std::string full_path = cifar_dataset_dir + "data_batch_1.bin";

                int noOfImages = 512;
                int cifarNoOfChannels = 3;
                int cifarImgWidth     = 32;
                int cifarImgHeight    = 32;

                int cifarImageDataSize = noOfImages * cifarNoOfChannels * cifarImgWidth * cifarImgHeight;
                float *h_cifarImageData   = (float*)calloc(cifarImageDataSize,   sizeof(float));  

                CifarReader::readCifarFile(full_path, noOfImages, h_cifarImageData);

            } 
    };
};

#endif