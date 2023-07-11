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

#include "SYCL.h"
#include "Utilities.h"
#include <iostream>

#define CPP_MODULE "SYCL"
#include "Logging.h"


cl::sycl::device SYCL::QueryAndGetDevice(std::string const &sDeviceType) 
{
    if (sDeviceType.empty()) {
        LOG("Using default selector()");
        return cl::sycl::device(cl::sycl::device(cl::sycl::default_selector()));
    }

    LOG("Attemping to use " << sDeviceType << "_selector()");
    std::string const sLowerCaseDeviceType(Utility::ToLowerCase(sDeviceType));
    if (sLowerCaseDeviceType == "cpu") 
        return cl::sycl::device(cl::sycl::device(cl::sycl::cpu_selector()));
    else if (sLowerCaseDeviceType == "gpu") 
       return cl::sycl::device(cl::sycl::device(cl::sycl::gpu_selector()));
    else if (sLowerCaseDeviceType == "acc") 
       return cl::sycl::device(cl::sycl::device(cl::sycl::accelerator_selector()));
    else if (sLowerCaseDeviceType == "host") 
       return cl::sycl::device(cl::sycl::device(cl::sycl::host_selector()));
    else if (sLowerCaseDeviceType == "nvidia")
        return cl::sycl::device(cl::sycl::device(nvidia_selector())); // Custom selector
    else {
        LOG_ERROR("Unknown device " << sDeviceType << "_selector()");
    }
}

SYCL::SYCL(cl::sycl::device const &SelectedDevice)
{
    std::function<void (cl::sycl::exception_list ExceptionList)> lQExceptionHandler([&](cl::sycl::exception_list ExceptionList) { // lambda Q exception handler
        try { 
            for (auto const &Exception : ExceptionList) {
                std::rethrow_exception(Exception);
            }
        } catch (cl::sycl::exception e) {
           LOG_ERROR("SYCL exception caught: " << e.what()); 
        }});

    m_sySelectedQueue = cl::sycl::queue(SelectedDevice, lQExceptionHandler, {cl::sycl::property::queue::enable_profiling()});
    DisplayDeviceProperties();
}

void SYCL::DisplayDeviceProperties() const
{
    LOG("SYCL device initialization successful")
    LOG("\t Using SYCL device         : " << m_sySelectedQueue.get_device().get_info<cl::sycl::info::device::name>() << 
        " (Driver version "     << m_sySelectedQueue.get_device().get_info<cl::sycl::info::device::driver_version>() << ")");

    std::vector<std::string> const vLDDPaths(Utility::ExtractLDDPathNameFromProcess({"libOpenCL", "libsycl", "libComputeCpp"})); //[0] OCL, [1] Intel's SYCL, [2] ComputeCpp SYCL
    if (vLDDPaths.empty()) {
        LOG_WARNING("Unable to print OpenCL and SYCL dependent libraries! The LD_LIBRARY_PATH may be incorrectly set"); // Should not reach to this case
        return;
    }

    LOG("\t Using OpenCL library      : " << (!vLDDPaths[0].empty() ? vLDDPaths[0] : "WARNING! OpenCL library not found!"));
   
    if (!vLDDPaths[1].empty())
       LOG("\t Using Intel's SYCL library: " << vLDDPaths[1]);

    if (!vLDDPaths[2].empty())
        LOG("\t Found ComputeCPP library  : " << vLDDPaths[2]);
}

