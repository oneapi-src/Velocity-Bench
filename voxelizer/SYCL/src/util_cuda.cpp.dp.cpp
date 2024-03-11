// Modifications Copyright (C) 2023 Intel Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
// OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
// OR OTHER DEALINGS IN THE SOFTWARE.

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
// #include "util_cuda.h"
#include "infra/infra.hpp"

// Check if SYCL requirements are met
bool initCuda(const sycl::queue &sycl_device_queue)
try
{
        int device_count = 0;

        // Is there a SYCL device at all?
        device_count = infra::dev_mgr::instance().device_count();
        if (device_count < 1)
        {
                fprintf(stderr, "[SYCL] No SYCL devices found. Make sure SYCL device is powered, connected and available. \n \n");
                fprintf(stderr, "[SYCL] On laptops: disable powersave/battery mode. \n");
                fprintf(stderr, "[SYCL] Exiting... \n");
                return false;
        }

        fprintf(stderr, "[SYCL] SYCL device(s) found \n");
        fprintf(stdout, "[SYCL] \n");
        int device = 0; // setting device to a default value 0.
        std::string deviceName = sycl_device_queue.get_device().get_info<cl::sycl::info::device::name>();
        fprintf(stdout, "[SYCL] device: %s \n", deviceName.c_str());
        return true;
}
catch (sycl::exception const &exc)
{
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                  << ", line:" << __LINE__ << std::endl;
        std::exit(1);
}