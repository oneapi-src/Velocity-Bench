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


#ifndef SYCL_CLASS_H
#define SYCL_CLASS_H

#include "CommandLineParser.h"

#include <sycl/sycl.hpp>
#include <string>
#include <vector>

class SYCL
{
private:
    cl::sycl::queue m_syclQueue;
    cl::sycl::event m_syclEvent;
public:
    // NVIDIA GPU selector
    class nvidia_selector : public cl::sycl::device_selector
    {
        virtual int operator()(cl::sycl::device const &SYCLDevice) const override 
        {
            if (!SYCLDevice.is_gpu())
                return -1;

            std::string const sVendor(SYCLDevice.get_info<cl::sycl::info::device::vendor>());
            return sVendor.find("NVIDIA") != std::string::npos ? 1000 : -1;
        }
    };

    explicit SYCL(cl::sycl::device const &Device);
            ~SYCL(){}

    SYCL           (SYCL const &RHS) = delete; 
    SYCL &operator=(SYCL const &RHS) = delete;

    cl::sycl::queue &GetQueue() { return m_syclQueue; } 
    cl::sycl::event &GetEvent() { return m_syclEvent; }

    static void             DisplayDeviceProperties(cl::sycl::device const &Device);
    static cl::sycl::device QueryAndGetDevice      (std::string const &sDeviceType);

    static std::chrono::steady_clock::duration ConvertKernelTimeToDuration(cl::sycl::event const &Event);

#ifdef USE_INFRASTRUCTURE // Move to testbench base??
    static void RegisterDeviceSettings (VelocityBench::CommandLineParser &cmdLineParser);
#endif
};

#endif
