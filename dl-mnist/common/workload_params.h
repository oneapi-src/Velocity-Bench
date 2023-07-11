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

#ifndef DL_MNIST_WORKLOAD_CUDA_H
#define DL_MNIST_WORKLOAD_CUDA_H

#include "TestBenchBase.h"
#include <string>

using namespace std;

namespace dl_infra {
    namespace common {
        class WorkloadParams : public TestBenchBase 
        {
            public:
                enum TensorMemPolicy {
                    ALL_MEM_ALLOC_AT_START,           //all tensors are allocated in memory at the beginning
                    MEM_ALLOC_DEALLOC_EVERY_CONV      //for each layer, tensors are allocated before convolution and deallocated after
                };

                enum ConvAlgo {
                    CUDNN_IMPLICIT_GEMM,
                    CUDNN_IMPLICIT_PRECOMP_GEMM,
                    CUDNN_GEMM,
                    CUDNN_DIRECT,
                    CUDNN_FFT,
                    CUDNN_FFT_TILING,
                    CUDNN_WINOGRAD,
                    CUDNN_WINOGRAD_NONFUSED,
                    CUDNN_FIND_BEST_ALGO,

                    ONEDNN_DIRECT,
                    ONEDNN_AUTO,
                    ONEDNN_WINOGRAD,

                    MIOPEN_IMPLICIT_GEMM,
                    MIOPEN_GEMM,
                    MIOPEN_DIRECT,
                    MIOPEN_FFT,
                    MIOPEN_WINOGRAD,
                    MIOPEN_FIND_BEST_ALGO                
                };

                enum DatasetReaderFormat {
                    NCHW,
                    NHWC
                };

                enum OneDnnConvPdMemFormat {
                    ONEDNN_CONVPD_NCHW,
                    ONEDNN_CONVPD_ANY
                };

            private:
                bool trace_func_boundaries, trace_conv_ops, trace_mem_ops;
                TensorMemPolicy tensorMemPolicy;
                ConvAlgo convAlgo;
                DatasetReaderFormat datasetReaderFormat;
                bool dryRun;
                OneDnnConvPdMemFormat oneDnnConvPdMemFormat;
                int noOfIterations;

            public:
                WorkloadParams(int argc, const char** argv);
                ~WorkloadParams(){;} 

                //WorkloadParams           (WorkloadParams const &RHS) = delete;
                //WorkloadParams &operator=(WorkloadParams const &RHS) = delete;
                void initializeCmdParser();
                
                bool is_trace_func_boundaries() { return trace_func_boundaries;}
                bool is_trace_conv_ops() { return trace_conv_ops;}
                bool is_trace_mem_ops() { return trace_mem_ops;}
                TensorMemPolicy getTensorMemPolicy() {return tensorMemPolicy;}
                ConvAlgo getConvAlgo() { return convAlgo; }
                DatasetReaderFormat getDatasetReaderFormat() { return datasetReaderFormat; }
                bool getDryRun() { return dryRun; }
                OneDnnConvPdMemFormat getOneDnnConvPdMemFormat() { return oneDnnConvPdMemFormat; }
                int getNoOfIterations() { return noOfIterations; }

            private:
                void parseInputParameters(const VelocityBench::CommandLineParser& parser);
                void parseTrace(string trace);
                void parseTensorMgmt(string tensorMgmtStr);
                void parseConvAlgo(string convAlgoStr);
                void parseDatasetReaderFormat(string datasetReaderFormatStr);
                void parseDryRun(string dryRunStr);
                void parseOneDnnConvPdMemFormat(string onednnConvPdMemFormatStr);
                void parseNoOfIterations(string noOfIterationsStr);
                
        };
    }
}
#endif


