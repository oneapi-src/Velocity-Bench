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

#ifndef DL_CIFAR_CAIT_H_
#define DL_CIFAR_CAIT_H_



#include <vector>
#include <cmath>
#include "timing.h"
#include "blas_routines.h"
#include "mlp.h"
#include "cifar_reader.h"
#include "image_processing.h"
#include "cait_self_attn_stage.h"
#include "cait_class_attn_stage.h"
#include "cait_configs.h"
#include "tracing.h"
#include "handle.h"
#include "workload_params.h"



namespace dl_cifar::common {
    class Cait {
        private:
            Timer* timer_;
            LangHandle *langHandle_;
            CaitParams caitParams_;

            CaitClassAttnStage *classAttentionStage_;
            CaitSelfAttnStage  *selfAttentionStage_;

            int noOfSAEncoders_, noOfCAEncoders_;

            int noOfPatches_;
            int imgsSize_;
            int patchEmbsSize_;
            int clsEmbSize_;

            float *d_inputImgs_, *d_imgPatches_, *d_d_imgPatches_, *d_patchEmbs_, *h_d_patchEmbs_, *d_d_patchEmbs_;
            float *h_d_clsEmb_, *d_clsEmb_, *d_d_clsEmb_;
            Mlp *patchToEmb_mlp_; 

        public:
            Cait(LangHandle *langHandle, Timer* timer, CaitParams caitParams, float *d_inputImgs);

            void doFw();
            void doBw();

            ~Cait();


    };

    class CaitController {
        public:
            static void execute(DlCifarWorkloadParams::DlNwSizeType dlNwSizeType, LangHandle *langHandle, Timer* timer, Timer* dataFileReadTimer) {
                
                CaitParams selectedCaitParams; 
                if(dlNwSizeType==DlCifarWorkloadParams::DlNwSizeType::LOW_MEM_GPU_SIZE) {
                    selectedCaitParams = CaitConfigs::caitM36_224_lowmemGPU;
                } else if(dlNwSizeType==DlCifarWorkloadParams::DlNwSizeType::WORKLOAD_DEFAULT_SIZE) {
                    selectedCaitParams = CaitConfigs::caitM36_224_workload_default;
                } else if(dlNwSizeType==DlCifarWorkloadParams::DlNwSizeType::FULL_SIZE) {
                    selectedCaitParams = CaitConfigs::caitM36_224_fullsize;
                } 
                
                // --------------- creating host and device mem for Cifar image data to be read into ----------------------------------------            
                int cifarRawImgsSize = selectedCaitParams.batchSize * CaitConfigs::cifarNoOfChannels * CaitConfigs::cifarImgWidth * CaitConfigs::cifarImgHeight;
                float *h_cifarRawImgs   = (float*)calloc(cifarRawImgsSize,   sizeof(float));  
                float *d_cifarRawImgs;
                d_cifarRawImgs = langHandle->allocDevMem((cifarRawImgsSize) * sizeof(float));
                
                // CIFAR images need to be resized to Cait M-36 224 image sizes
                // --------------- creating device mem for resized image data ----------------------------------------
                int resizedSize = selectedCaitParams.batchSize * selectedCaitParams.imgNoOfChannels * selectedCaitParams.imgWidth * selectedCaitParams.imgHeight;
                float *h_resizedImgs   = (float*)calloc(resizedSize,   sizeof(float));  
                float *d_resizedImgs;
                d_resizedImgs = langHandle->allocDevMem((resizedSize) * sizeof(float));


                Cait *cait = new Cait(langHandle, timer, selectedCaitParams, d_resizedImgs);


            

                // ----------------- train --------------------------
                int iterations = 1; 
                for(int i=0; i<iterations; i++) {
                    //read Cifar image data, then resize and finally do a caitM36 FW pass 
                    Time start = get_time_now();
                    CifarReader::readCifarFile(CaitConfigs::cifar_dataset_file, selectedCaitParams.batchSize, h_cifarRawImgs);
                    dataFileReadTimer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "READ_CIAR_FILE");

                    langHandle->memCpyH2D(d_cifarRawImgs, h_cifarRawImgs, sizeof(float) * cifarRawImgsSize, true);

                    ImageProcessor::resize(langHandle, d_cifarRawImgs, d_resizedImgs, selectedCaitParams.batchSize, 
                                            CaitConfigs::cifarNoOfChannels, CaitConfigs::cifarImgWidth, CaitConfigs::cifarImgHeight, 
                                            selectedCaitParams.imgWidth, selectedCaitParams.imgHeight); 

                    // ImageProcessor::resizeInHost(langHandle, h_cifarRawImgs, h_resizedImgs, selectedCaitParams.batchSize, 
                    //                         CaitConfigs::cifarNoOfChannels, CaitConfigs::cifarImgWidth, CaitConfigs::cifarImgWidth, 
                    //                         selectedCaitParams.imgWidth, selectedCaitParams.imgHeight); 
                    // langHandle->memCpyH2D(d_resizedImgs, h_resizedImgs, sizeof(float) * resizedSize, true);



                    cait->doFw();

                    cait->doBw();
                }

                free(h_cifarRawImgs);
                free(h_resizedImgs);
                langHandle->freeDevPtr(d_cifarRawImgs);
                langHandle->freeDevPtr(d_resizedImgs);
                delete cait;
 
            }
    };
};
#endif