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

#ifndef DL_CIFAR_WORKLOAD_CUDA_H
#define DL_CIFAR_WORKLOAD_CUDA_H
#include "TestBenchBase.h"

namespace dl_cifar::common {
    class DlCifarWorkloadParams : public TestBenchBase 
    {
        public:
            enum DlNwSizeType {
                LOW_MEM_GPU_SIZE,                // for GPUs with low memory; useful for debugging too
                WORKLOAD_DEFAULT_SIZE,           // default size 
                FULL_SIZE                   // typically will not fit in single GPU memory
            };

        private:
            bool trace_func_boundaries, trace_conv_ops, trace_mem_ops;
            DlNwSizeType dlNwSizeType;
            int32_t emb_len, max_seq_len, heads_count, qkv_len;
        public:
            DlCifarWorkloadParams(int argc, const char** argv);
            ~DlCifarWorkloadParams(){;} 
            void initializeCmdParser();

            bool is_trace_func_boundaries() { return trace_func_boundaries;}
            bool is_trace_conv_ops() { return trace_conv_ops;}
            bool is_trace_mem_ops() { return trace_mem_ops;}

            DlNwSizeType getDlNwSizeType() {return dlNwSizeType;}
            

        private:
            void parseInputParameters(const VelocityBench::CommandLineParser& parser);
            void parseTrace(std::string trace);
            void parseDlNwSizeType(std::string dlNwSizeTypeStr);
    };
};


#endif

