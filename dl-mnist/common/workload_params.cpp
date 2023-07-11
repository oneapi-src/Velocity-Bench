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

#include "CommandLineParser.h"
#include "TestBenchBase.h"
#include "workload_params.h"
#include "utils.h"
#include "tracer.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#define CPP_MODULE "WL PARAMS"
#include "Logging.h"
#include "Timer.h"

#ifndef DEBUG
#define DEBUG 0 
#endif

using namespace std;
using namespace dl_infra::common;

bool Tracer::trace_func_boundaries = false;
bool Tracer::trace_conv_op = false;
bool Tracer::trace_mem_op = false;

namespace dl_infra {
    namespace common {
        WorkloadParams::WorkloadParams(int argc, const char** argv)
            : TestBenchBase()
        {
            bool bRequired(true);
            VelocityBench::CommandLineParser &parser = GetCmdLineParser();
            parser.AddSetting("-trace", "Trace type",                     !bRequired, "notrace",  VelocityBench::CommandLineParser::InputType_t::STRING, 1);
            parser.AddSetting("-tensor_mgmt", "Tensor management policy", !bRequired, "per_layer", VelocityBench::CommandLineParser::InputType_t::STRING, 1);
            parser.AddSetting("-conv_algo", "Convolution algorithm", bRequired, "", VelocityBench::CommandLineParser::InputType_t::STRING, 1);
            parser.AddSetting("-dataset_reader_format", "Dataset reader format", !bRequired, "NCHW", VelocityBench::CommandLineParser::InputType_t::STRING, 1);
            parser.AddSetting("-dry_run", "Dry run", !bRequired, "YES", VelocityBench::CommandLineParser::InputType_t::STRING, 1);
            parser.AddSetting("-onednn_convpd_mem_format", "OneDNN Conv PD memory format", !bRequired, "ONEDNN_CONVPD_ANY", VelocityBench::CommandLineParser::InputType_t::STRING, 1);
            parser.AddSetting("-iteration_count", "No of iterations for inference", !bRequired, "400", VelocityBench::CommandLineParser::InputType_t::INTEGER, 1);

            ParseCommandLineArguments(argc, argv);
            parseInputParameters(parser);    
        }

        void WorkloadParams::parseTrace(string trace) {
            stringstream ss(trace);
            vector<string> traceConfig;
            Tracer tracer;
            //bool Tracer::trace_func_boundarie = false;
            while( ss.good() )
            {
                string substr;;
                getline( ss, substr, ',' );
                traceConfig.push_back( substr );
            }

            if (std::find(traceConfig.begin(), traceConfig.end(), "func") != traceConfig.end()) {
                Tracer::set_trace_func_boundaries();
            } else {
                Tracer::unset_trace_func_boundaries();        
            }
            if (std::find(traceConfig.begin(), traceConfig.end(), "conv") != traceConfig.end()) {
                Tracer::set_trace_conv_op();
            } else {
                Tracer::unset_trace_conv_op();    
            } 
            if (std::find(traceConfig.begin(), traceConfig.end(), "mem") != traceConfig.end()) {
                Tracer::set_trace_mem_op();
            } else {
                Tracer::unset_trace_mem_op();    
            }
        }

        void WorkloadParams::parseTensorMgmt(string tensorMgmtStr) {
            if(tensorMgmtStr.compare("per_layer") == 0) {
                tensorMemPolicy = MEM_ALLOC_DEALLOC_EVERY_CONV;
            } else if(tensorMgmtStr.compare("prealloc_all") == 0) {
                tensorMemPolicy = ALL_MEM_ALLOC_AT_START;
            } else {
                LOG_ERROR("Invalid value for tensor_mgmt. Should be either 'per_layer' or 'prealloc_all'");
            }    
        }

        void WorkloadParams::parseOneDnnConvPdMemFormat(string oneDnnConvPdMemFormatStr) {
            if(oneDnnConvPdMemFormatStr.compare("ONEDNN_CONVPD_NCHW") == 0) {
                oneDnnConvPdMemFormat = ONEDNN_CONVPD_NCHW;
            } else if(oneDnnConvPdMemFormatStr.compare("ONEDNN_CONVPD_ANY") == 0) {
                oneDnnConvPdMemFormat = ONEDNN_CONVPD_ANY;
            } else {
                LOG_ERROR("Invalid value for onednn_convpd_mem_format. Should be either 'ONEDNN_NCHW' or 'ONEDNN_ANY'");
            }    
        }

        void WorkloadParams::parseDryRun(string dryRunStr) {
            if(dryRunStr.compare("YES") == 0) {
                dryRun = true;
            } else if(dryRunStr.compare("NO") == 0) {
                dryRun = false;
            } else {
                LOG_ERROR("Invalid value for dry_run. Should be either 'DRY_RUN' or 'NO_DRY_RUN'");
            }    
        }

        void WorkloadParams::parseConvAlgo(string convAlgoStr) {
            if(convAlgoStr.compare("CUDNN_IMPLICIT_GEMM") == 0) {
                convAlgo = CUDNN_IMPLICIT_GEMM;
            } else if(convAlgoStr.compare("CUDNN_IMPLICIT_PRECOMP_GEMM") == 0) {
                convAlgo = CUDNN_IMPLICIT_PRECOMP_GEMM;
            } else if(convAlgoStr.compare("CUDNN_GEMM") == 0) {
                convAlgo = CUDNN_GEMM;
            } else if(convAlgoStr.compare("CUDNN_DIRECT") == 0) {
                convAlgo = CUDNN_DIRECT;
            } else if(convAlgoStr.compare("CUDNN_FFT") == 0) {
                convAlgo = CUDNN_FFT;
            } else if(convAlgoStr.compare("CUDNN_FFT_TILING") == 0) {
                convAlgo = CUDNN_FFT_TILING;
            } else if(convAlgoStr.compare("CUDNN_WINOGRAD") == 0) {
                convAlgo = CUDNN_WINOGRAD;
            } else if(convAlgoStr.compare("CUDNN_WINOGRAD_NONFUSED") == 0) {
                convAlgo = CUDNN_WINOGRAD_NONFUSED;
            } else if(convAlgoStr.compare("ONEDNN_DIRECT") == 0) {
                convAlgo = ONEDNN_DIRECT;
            } else if(convAlgoStr.compare("ONEDNN_AUTO") == 0) {
                convAlgo = ONEDNN_AUTO;
            } else if(convAlgoStr.compare("ONEDNN_WINOGRAD") == 0) {
                convAlgo = ONEDNN_WINOGRAD;
            } else if(convAlgoStr.compare("CUDNN_FIND_BEST_ALGO") == 0) {
                convAlgo = CUDNN_FIND_BEST_ALGO;
            } else if(convAlgoStr.compare("MIOPEN_IMPLICIT_GEMM") == 0) {
                convAlgo = MIOPEN_IMPLICIT_GEMM;
            } else if(convAlgoStr.compare("MIOPEN_GEMM") == 0) {
                convAlgo = MIOPEN_GEMM;
            } else if(convAlgoStr.compare("MIOPEN_DIRECT") == 0) {
                convAlgo = MIOPEN_DIRECT;
            } else if(convAlgoStr.compare("MIOPEN_FFT") == 0) {
                convAlgo = MIOPEN_FFT;
            } else if(convAlgoStr.compare("MIOPEN_WINOGRAD") == 0) {
                convAlgo = MIOPEN_WINOGRAD;
            } else if(convAlgoStr.compare("MIOPEN_FIND_BEST_ALGO") == 0) {
                convAlgo = MIOPEN_FIND_BEST_ALGO;
            } else {
                LOG_ERROR("Invalid value for conv_algo.");
            }    
        }

        void WorkloadParams::parseDatasetReaderFormat(string datasetReaderFormatStr) {
            if(datasetReaderFormatStr.compare("NCHW") == 0) {
                datasetReaderFormat = NCHW;
            } else if(datasetReaderFormatStr.compare("NHWC") == 0) {
                datasetReaderFormat = NHWC;
            } else {
                LOG_ERROR("Invalid value for tensor_mgmt. Should be either 'per_layer' or 'prealloc_all'");
            }    
        }

        void WorkloadParams::parseNoOfIterations(string noOfIterationsStr) {
            noOfIterations = std::stoi(noOfIterationsStr);
        }

        void WorkloadParams::parseInputParameters(const VelocityBench::CommandLineParser& parser) {
            string traceStr                 = parser.GetSetting("-trace");
            string tensorMgmtStr            = parser.GetSetting("-tensor_mgmt");
            string convAlgoStr              = parser.GetSetting("-conv_algo");
            string datasetReaderFormatStr   = parser.GetSetting("-dataset_reader_format");
            string dryRunStr                = parser.GetSetting("-dry_run");
            string onednnConvPdMemFormatStr = parser.GetSetting("-onednn_convpd_mem_format");
            string noOfIterationsStr        = parser.GetSetting("-iteration_count");
            
            LOG( "\n");
            LOG( "==================================================");
            LOG("User input parameters:"); 
            LOG("Trace:                             "   << traceStr);
            LOG("Tensor management policy:          "   << tensorMgmtStr);
            LOG("Convolution algorithm:             "   << convAlgoStr);
            LOG("Dataset reader format:             "   << datasetReaderFormatStr);
            LOG("Dry run:                           "   << dryRunStr);
            LOG("OneDNN Conv PD memory format:      "   << onednnConvPdMemFormatStr);
            LOG("No of iterations for inference:    "   << noOfIterationsStr);
            LOG( "==================================================");
            LOG( "\n");

            parseTrace(traceStr);
            parseTensorMgmt(tensorMgmtStr);
            parseConvAlgo(convAlgoStr);
            parseDatasetReaderFormat(datasetReaderFormatStr);
            parseDryRun(dryRunStr);
            parseOneDnnConvPdMemFormat(onednnConvPdMemFormatStr);
            parseNoOfIterations(noOfIterationsStr);
        }
    }
}
