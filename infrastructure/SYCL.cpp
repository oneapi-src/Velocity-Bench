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


#include "SYCL.h"
#include "Utilities.h"
#include <iostream>
#include <numeric>

#define CPP_MODULE "SYCL"
#include "Logging.h"

#ifdef USE_INFRASTRUCTURE // Move to test bench base??
void SYCL::RegisterDeviceSettings(VelocityBench::CommandLineParser &cmdLineParser) 
{
    cmdLineParser.AddSetting("--device_type", "SYCL device type to use", false, "", VelocityBench::CommandLineParser::InputType_t::STRING, 1 /* One operand */);
    LOG("Registered SYCL device querying settings");
}
#endif

sycl::device SYCL::QueryAndGetDevice(std::string const &sDeviceType) 
{
    if (sDeviceType.empty()) {
        LOG("Using default selector()");
        return sycl::device(sycl::default_selector_v);
    }

    LOG("Attemping to use " << sDeviceType << "_selector()");
    std::string const sLowerCaseDeviceType(Utility::ToLowerCase(sDeviceType));
    if (sLowerCaseDeviceType == "cpu") 
        return sycl::device(sycl::cpu_selector_v);
    else if (sLowerCaseDeviceType == "gpu") 
       return sycl::device(sycl::gpu_selector_v);
    else {
        LOG_ERROR("Unknown device " << sDeviceType << "_selector()");
    }
}

SYCL::SYCL(sycl::device const &SelectedDevice, bool const /*bUseInOrderQueue*/)
    : m_syclQueue(SelectedDevice,                                       // Device selected
                  [](sycl::exception_list ExceptionList) {          // Async exception handler
                  try { 
                      for (auto const &Exception : ExceptionList) {
                          std::rethrow_exception(Exception);
                      }
                  } catch (sycl::exception e) {
                     LOG_ERROR("SYCL exception caught: " << e.what()); 
                  }}
                  , {sycl::property::queue::in_order()}             // Use in order queue 
                  )
    , m_syclEvent()
{
    LOG("SYCL Queue initialization successful ");
    DisplayProperties();
}

SYCL::SYCL(sycl::device const &SelectedDevice)
    : m_syclQueue(SelectedDevice,                                       // Device selected
                  [](sycl::exception_list ExceptionList) {          // Async exception handler
                  try { 
                      for (auto const &Exception : ExceptionList) {
                          std::rethrow_exception(Exception);
                      }
                  } catch (sycl::exception e) {
                     LOG_ERROR("SYCL exception caught: " << e.what()); 
                  }}
#ifdef ENABLE_KERNEL_PROFILING
                  , {sycl::property::queue::enable_profiling()}     // Setting profiling if we want to
#endif
                  )
    , m_syclEvent()
{
    LOG("SYCL Queue initialization successful");
    DisplayProperties();
}

void SYCL::DisplayProperties() //sycl::device const &Device)
{
    LOG("\t Using SYCL device         : " << m_syclQueue.get_device().get_info<sycl::info::device::name>() << " (Driver version " << m_syclQueue.get_device().get_info<sycl::info::device::driver_version>() << ")");
    LOG("\t Platform                  : " << m_syclQueue.get_device().get_platform().get_info<sycl::info::platform::name>());
    LOG("\t Vendor                    : " << m_syclQueue.get_device().get_info<sycl::info::device::vendor>());
    LOG("\t Max compute units         : " << m_syclQueue.get_device().get_info<sycl::info::device::max_compute_units>());
    LOG("\t Queue type                : " << (m_syclQueue.is_in_order() ? "In order" : "Out of order"));

    std::vector<std::string> const vLDDPaths(Utility::ExtractLDDPathNameFromProcess({"libOpenCL", "libsycl", "libComputeCpp", "libze"})); //[0] OCL, [1] Intel's SYCL, [2] ComputeCpp SYCL
    if (vLDDPaths.empty()) {
        LOG_WARNING("Unable to print OpenCL and SYCL dependent libraries! The LD_LIBRARY_PATH may be incorrectly set"); // Should not reach to this case
        return;
    }

    LOG("\t Using OpenCL library      : " << (!vLDDPaths[0].empty() ? vLDDPaths[0] : "WARNING! OpenCL library not found!"));
   
    if (!vLDDPaths[1].empty()) { // Implies we are using Intel's DPC++ compiler
       LOG("\t Using OneAPI SYCL library : " << vLDDPaths[1]);
       LOG("\t Using Level Zero library  : " << (!vLDDPaths[3].empty() ? vLDDPaths[3] : "WARNING! Level zero library not found! L0 backend may not be available!"));
    }

    if (!vLDDPaths[2].empty())
       LOG("\t Using ComputeCPP library  : " << vLDDPaths[2]);
}

std::chrono::steady_clock::duration SYCL::ConvertKernelTimeToDuration(sycl::event const &Event)
{
    std::chrono::time_point<std::chrono::steady_clock> const tpStart (std::chrono::nanoseconds(Event.template get_profiling_info<sycl::info::event_profiling::command_start>())); 
    std::chrono::time_point<std::chrono::steady_clock> const tpFinish(std::chrono::nanoseconds(Event.template get_profiling_info<sycl::info::event_profiling::command_end>())); 
    return std::chrono::steady_clock::duration(tpFinish - tpStart);
}
