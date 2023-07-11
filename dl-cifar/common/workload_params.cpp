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
#include "tracing.h"
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

bool dl_cifar::common::Tracer::trace_func_boundaries = false;
bool dl_cifar::common::Tracer::trace_conv_op = false;
bool dl_cifar::common::Tracer::trace_mem_op = false;

namespace dl_cifar::common {
    DlCifarWorkloadParams::DlCifarWorkloadParams(int argc, const char** argv)
        : TestBenchBase()
    {
        bool bRequired(true);
        VelocityBench::CommandLineParser &parser = GetCmdLineParser();

        parser.AddSetting("-trace", "Trace type",                      !bRequired, "notrace",  VelocityBench::CommandLineParser::InputType_t::STRING, 1);
        parser.AddSetting("-dl_nw_size_type", "DL NW size type",       !bRequired, "WORKLOAD_DEFAULT_SIZE", VelocityBench::CommandLineParser::InputType_t::STRING, 1);
        
        ParseCommandLineArguments(argc, argv);
        parseInputParameters(parser);    
    }


    void DlCifarWorkloadParams::parseTrace(std::string trace) {
        std::stringstream ss(trace);
        std::vector<std::string> traceConfig;
        Tracer tracer;
        //bool Tracer::trace_func_boundarie = false;
        while( ss.good() )
        {
            std::string substr;;
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

    void DlCifarWorkloadParams::parseInputParameters(const VelocityBench::CommandLineParser& parser) {
        std::string traceStr = parser.GetSetting("-trace");
        std::string dlNwSizeTypeStr     =  parser.GetSetting("-dl_nw_size_type"); 

        LOG( "\n");
        LOG( "==================================================");
        LOG("User input parameters:"); 
        LOG("Trace:                         " << traceStr);
        LOG("DL NW size type:               " << dlNwSizeTypeStr);
        LOG( "==================================================");
        LOG( "\n");

        parseTrace(traceStr);
        parseDlNwSizeType(dlNwSizeTypeStr);
    }

    void DlCifarWorkloadParams::parseDlNwSizeType(std::string dlNwSizeTypeStr) {
            if(dlNwSizeTypeStr.compare("LOW_MEM_GPU_SIZE") == 0) {
                dlNwSizeType = LOW_MEM_GPU_SIZE;
            } else if(dlNwSizeTypeStr.compare("WORKLOAD_DEFAULT_SIZE") == 0) {
                dlNwSizeType = WORKLOAD_DEFAULT_SIZE;
            } else if(dlNwSizeTypeStr.compare("FULL_SIZE") == 0) {
                dlNwSizeType = FULL_SIZE;
            } else {
                LOG_ERROR("Invalid value for dl_nw_size_type. Should be one of 'LOW_MEM_GPU_SIZE', 'WORKLOAD_DEFAULT_SIZE' or 'FULL_SIZE'");
            }    
        }
};