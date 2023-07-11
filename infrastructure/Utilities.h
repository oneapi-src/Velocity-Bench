/*
 * 
 * Modifications Copyright (C) 2023 Intel Corporation
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * 
 * of this software and associated documentation files (the "Software"),
 * 
 * to deal in the Software without restriction, including without limitation
 * 
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * 
 * and/or sell copies of the Software, and to permit persons to whom
 * 
 * the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * 
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * 
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * 
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * 
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * 
 * OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 * SPDX-License-Identifier: MIT
 */


#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <random>
#include <functional>

namespace Utility
{
    bool IsHTEnabled();
    bool IsHTEnabled(std::string const &sParentCPUName);
    bool IsInteger  (std::string const &sStringToCheck);

    std::string ToUpperCase(std::string const &sStringToTransform);
    std::string ToLowerCase(std::string const &sStringToTransform);
    std::string GetCurrentRunningDirectory();
    std::string GetHostName();
    std::string GetProcessID();
    std::string GetEnvironmentValue(std::string const &sEnvironmentName);
    std::string DisplayHumanReadableBytes(size_t const uBytes);
    std::string ConvertTimeToReadable(double const tTimeInNanoSeconds);

    std::vector<std::string> ExtractLDDPathNameFromProcess(std::vector<std::string> const &vLDDPathsToSearch);
    std::vector<std::string> TokenizeString(std::string const &sStringToTokenize, char const cDelimiter);

    float RandomFloatNumber  (double const dBeginRange, double const dEndRange);
    int   RandomIntegerNumber(int    const iBeginRange, int    const iEndRange);
}

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

namespace Utility
{
    void QueryCUDADevice(); 
    void PrintCUDADeviceProperty(const cudaDeviceProp& prop);
}
#elif USE_HIP
#include <hip/hip_runtime.h>
namespace Utility
{
    void QueryHIPDevice(); 
    void PrintHIPDeviceProperty(hipDeviceProp_t const &prop);
}

#endif


#endif
