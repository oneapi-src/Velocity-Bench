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

#ifndef CUDASIFT_WL_PARAMS_H
#define CUDASIFT_WL_PARAMS_H

#include "TestBenchBase.h"
#include <string>

using namespace std;

namespace common {
    class WorkloadParams : public TestBenchBase 
    {
        public:

        private:
            string inputDataLoc;

        public:
            WorkloadParams(int argc, const char** argv);
            ~WorkloadParams(){;} 

            //WorkloadParams           (WorkloadParams const &RHS) = delete;
            //WorkloadParams &operator=(WorkloadParams const &RHS) = delete;
            void initializeCmdParser();
            string getInputDataLoc() { return inputDataLoc;}

        private:
            void parseInputParameters(const VelocityBench::CommandLineParser& parser);

    };
}

#endif
