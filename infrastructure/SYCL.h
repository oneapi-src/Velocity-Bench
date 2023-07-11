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


#ifndef SYCL_CLASS_H
#define SYCL_CLASS_H

#include "CommandLineParser.h"

#include <sycl.hpp>
#include <string>
#include <vector>

class SYCL
{
private:
    sycl::queue m_syclQueue; // contains device and context info
    sycl::event m_syclEvent;
public:
    explicit SYCL(sycl::device const &Device);

    SYCL(sycl::device const &Device, bool const bUseInOrderQueue);
   ~SYCL(){}

    SYCL           (SYCL const &RHS) = delete; 
    SYCL &operator=(SYCL const &RHS) = delete;

    sycl::queue  &GetQueue() { return m_syclQueue; }
    sycl::event  &GetEvent() { return m_syclEvent; }

    sycl::device  const GetDevice()  { return m_syclQueue.get_device();  }
    sycl::context const GetContext() { return m_syclQueue.get_context(); }

    void DisplayProperties();
    static sycl::device QueryAndGetDevice(std::string const &sDeviceType);

    static std::chrono::steady_clock::duration ConvertKernelTimeToDuration(sycl::event const &Event);

#ifdef USE_INFRASTRUCTURE // Move to testbench base??
    static void RegisterDeviceSettings (VelocityBench::CommandLineParser &cmdLineParser);
#endif
};

#endif
