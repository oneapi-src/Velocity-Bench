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

#ifndef DL_CIFAR_VIT_H_
#define DL_CIFAR_VIT_H_

#include <vector>
#include <cmath>
#include "timing.h"
#include "mlp.h"
#include "vit_multi_headed_self_attn.h"
#include "vit_transformer.h"
#include "cifar_reader.h"
#include "image_processing.h"
#include "tracing.h"
#include "handle.h"
#include "vit_configs.h"
#include "workload_params.h"


namespace dl_cifar::common {
    class Vit {
        private:
            Timer* timer_;
            LangHandle *langHandle_;

            VitParams vitParams_;
            int noOfPatches_;
            int imgsSize_;
            int patchEmbsSize_;

            float *d_inputImgs_;
            float *d_imgPatches_, *d_d_imgPatches_; 
            float *d_patchEmbs_,  *d_d_patchEmbs_,  *h_d_patchEmbs_;
            Mlp *patchToEmb_mlp_; 

            VitTransformer *transformer_;
            

        public: 
            Vit(LangHandle *langHandle, Timer* timer, VitParams vitParams, float *d_inputImgs);

            void doFw();
            void doBw();

            ~Vit();

        
    };




    class VitController {
        public:

            static void execute(DlCifarWorkloadParams::DlNwSizeType dlNwSizeType, LangHandle *langHandle, Timer* timer, Timer* dataFileReadTimer) {
                
                VitParams selectedVitParams; 
                if(dlNwSizeType==DlCifarWorkloadParams::DlNwSizeType::LOW_MEM_GPU_SIZE) {
                    selectedVitParams = VitConfigs::vitL16_params_lowmemGPU;
                } else if(dlNwSizeType==DlCifarWorkloadParams::DlNwSizeType::WORKLOAD_DEFAULT_SIZE) {
                    selectedVitParams = VitConfigs::vitL16_params_workload_default;
                } else if(dlNwSizeType==DlCifarWorkloadParams::DlNwSizeType::FULL_SIZE) {
                    selectedVitParams = VitConfigs::vitL16_params_fullsize;
                } 

                int iterations = 1; 

                
                int noOfPatches = (selectedVitParams.imgWidth * selectedVitParams.imgHeight) / (selectedVitParams.patchSize * selectedVitParams.patchSize); 

                // --------------- creating host and device mem for Cifar image data to be read into ----------------------------------------            
                int cifarRawImgsSize = selectedVitParams.batchSize * VitConfigs::cifarNoOfChannels * VitConfigs::cifarImgWidth * VitConfigs::cifarImgWidth;
                float *h_cifarRawImgs   = (float*)calloc(cifarRawImgsSize,   sizeof(float));  
                float *d_cifarRawImgs = langHandle->allocDevMem((cifarRawImgsSize) * sizeof(float));
                
                // CIFAR images need to be resized to vit L16 image sizes
                // --------------- creating device mem for resized image data ----------------------------------------
                int resizedSize = selectedVitParams.batchSize * selectedVitParams.imgNoOfChannels * selectedVitParams.imgWidth * selectedVitParams.imgHeight;
                float *h_resizedImgs   = (float*)calloc(resizedSize,   sizeof(float));  
                float *d_resizedImgs = langHandle->allocDevMem((resizedSize) * sizeof(float));

                
                // --------------create Vit -----------------------

                Vit *vitL16 = new Vit(langHandle, timer, selectedVitParams, d_resizedImgs);

                //----------------- train --------------------------
                for(int i=0; i<iterations; i++) {
                    
                    //read Cifar image data, then resize and finally do a vitL16 FW pass 
                    Time start = get_time_now();
                    CifarReader::readCifarFile(VitConfigs::cifar_dataset_file, selectedVitParams.batchSize, h_cifarRawImgs);
                    dataFileReadTimer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "READ_CIAR_FILE");

                    langHandle->memCpyH2D(d_cifarRawImgs, h_cifarRawImgs, sizeof(float) * cifarRawImgsSize, true);
                    ImageProcessor::resize(langHandle, d_cifarRawImgs, d_resizedImgs, selectedVitParams.batchSize, 
                                            VitConfigs::cifarNoOfChannels, VitConfigs::cifarImgWidth, VitConfigs::cifarImgWidth, 
                                            selectedVitParams.imgWidth, selectedVitParams.imgHeight); 


                    // ImageProcessor::resizeInHost(langHandle, h_cifarRawImgs, h_resizedImgs, selectedVitParams.batchSize, 
                    //                         VitConfigs::cifarNoOfChannels, VitConfigs::cifarImgWidth, VitConfigs::cifarImgWidth, 
                    //                         selectedVitParams.imgWidth, selectedVitParams.imgHeight); 
                    // langHandle->memCpyH2D(d_resizedImgs, h_resizedImgs, sizeof(float) * resizedSize, true);


                    vitL16->doFw();
                    vitL16->doBw();
                }

                free(h_cifarRawImgs);
                free(h_resizedImgs);
                langHandle->freeDevPtr(d_cifarRawImgs);
                langHandle->freeDevPtr(d_resizedImgs);
                delete vitL16;








                if(dlNwSizeType==DlCifarWorkloadParams::DlNwSizeType::LOW_MEM_GPU_SIZE) {
                    selectedVitParams = VitConfigs::vitH14_params_lowmemGPU;
                } else if(dlNwSizeType==DlCifarWorkloadParams::DlNwSizeType::WORKLOAD_DEFAULT_SIZE) {
                    selectedVitParams = VitConfigs::vitH14_params_workload_default;
                } else if(dlNwSizeType==DlCifarWorkloadParams::DlNwSizeType::FULL_SIZE) {
                    selectedVitParams = VitConfigs::vitH14_params_fullsize;
                } 

                
                // --------------- creating host and device mem for Cifar image data to be read into ----------------------------------------            
                cifarRawImgsSize = selectedVitParams.batchSize * VitConfigs::cifarNoOfChannels * VitConfigs::cifarImgWidth * VitConfigs::cifarImgWidth;
                h_cifarRawImgs   = (float*)calloc(cifarRawImgsSize,   sizeof(float));  
                d_cifarRawImgs = langHandle->allocDevMem((cifarRawImgsSize) * sizeof(float));

                
                // --------------- creating device mem for resized image data ----------------------------------------
                resizedSize = selectedVitParams.batchSize * selectedVitParams.imgNoOfChannels * selectedVitParams.imgWidth * selectedVitParams.imgHeight;                
                h_resizedImgs   = (float*)calloc(resizedSize,   sizeof(float));  
                d_resizedImgs = langHandle->allocDevMem((resizedSize) * sizeof(float));

                
                // // --------------create Vit -----------------------

                Vit *vitH14 = new Vit(langHandle, timer, selectedVitParams, d_resizedImgs);

                // ----------------- train --------------------------
                for(int i=0; i<iterations; i++) {
                    //read Cifar image data, then resize and finally do a vitH14 FW pass 
                    Time start = get_time_now();
                    CifarReader::readCifarFile(VitConfigs::cifar_dataset_file, selectedVitParams.batchSize, h_cifarRawImgs);
                    dataFileReadTimer->recordOpTimeTaken(1000, calculate_op_time_taken(start), "READ_CIAR_FILE");
                    
                    

                    langHandle->memCpyH2D(d_cifarRawImgs, h_cifarRawImgs, sizeof(float) * cifarRawImgsSize, true);
                    ImageProcessor::resize(langHandle, d_cifarRawImgs, d_resizedImgs, selectedVitParams.batchSize, 
                                            VitConfigs::cifarNoOfChannels, VitConfigs::cifarImgWidth, VitConfigs::cifarImgWidth, 
                                            selectedVitParams.imgWidth, selectedVitParams.imgHeight); 
                    // ImageProcessor::resizeInHost(langHandle, h_cifarRawImgs, h_resizedImgs, selectedVitParams.batchSize,
                    //                         VitConfigs::cifarNoOfChannels, VitConfigs::cifarImgWidth, VitConfigs::cifarImgWidth,
                    //                         selectedVitParams.imgWidth, selectedVitParams.imgHeight);
                    // langHandle->memCpyH2D(d_resizedImgs, h_resizedImgs, sizeof(float) * resizedSize, true);



                    vitH14->doFw();
                    vitH14->doBw();
                }

                free(h_cifarRawImgs);
                free(h_resizedImgs);
                langHandle->freeDevPtr(d_cifarRawImgs);
                langHandle->freeDevPtr(d_resizedImgs);
                
                delete vitH14;

            }
    };
};



#endif