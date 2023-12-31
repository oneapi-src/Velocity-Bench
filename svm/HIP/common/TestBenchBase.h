/*
MIT License

Copyright (c) 2015 University of West Bohemia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
MIT License

Modifications Copyright (C) 2023 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

SPDX-License-Identifier: MIT License
*/


#ifndef TEST_BENCH_BASE_H
#define TEST_BENCH_BASE_H

#include "FileHandler.h"
#include "CommandLineParser.h"

#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>


#ifdef USE_OPENCL
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#elif  USE_SYCL
#include <sycl/sycl.hpp>
#elif  USE_CUDA
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#endif

class TestBenchBase
{
protected:
    CommandLineParser m_cmdLineParser;
    unsigned int GetNumberOfIterations() const;

public:
    TestBenchBase           (TestBenchBase const &RHS) = delete;
    TestBenchBase &operator=(TestBenchBase const &RHS) = delete;

    TestBenchBase();
   ~TestBenchBase(){} 

    CommandLineParser  &GetCmdLineParser()     { return m_cmdLineParser; } // Used for adding more settings to the application

    bool ParseCommandLineArguments(int const argc, char const *argv[]);
};

#endif


