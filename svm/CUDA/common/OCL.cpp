
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

#include "OCL.h"
#include "Utilities.h"

#define CPP_MODULE "OCLC"
#include "Logging.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <locale> // toUpper
#include <experimental/filesystem> // Real canonical path
#include <system_error>



std::vector<cl::Device> OCL::GetAllDevices() 
{
    std::vector<cl::Platform> vCLPlatforms;
    cl::Platform::get(&vCLPlatforms);
    LOG_ASSERT(!vCLPlatforms.empty(), Utility::GetHostName() << " has no platform available. Please use a machine with at least one supported OpenCL platform"); 

    std::vector<cl::Device> vAllDevices;
    for (auto const &CLPlatform : vCLPlatforms) {
        std::vector<cl::Device> vCurrentPlatformDevices;
        CLPlatform.getDevices(CL_DEVICE_TYPE_ALL, &vCurrentPlatformDevices);
        vAllDevices.insert(vAllDevices.end(), vCurrentPlatformDevices.begin(), vCurrentPlatformDevices.end());
    }

    LOG_ASSERT(!vAllDevices.empty(), Utility::GetHostName() << " has no devices available. Please use a machine with at least one supported OpenCL device");
    return vAllDevices;
}

cl::Device OCL::GetDevice(std::string const &sDeviceType)
{
    if (sDeviceType.empty())
        LOG_ERROR("Device type should not be empty!");

    LOG("Attempting to find device type " << sDeviceType);
    std::string const sLowerCaseDeviceType(Utility::ToLowerCase(sDeviceType));
    if (sLowerCaseDeviceType != "cpu" && sLowerCaseDeviceType != "gpu" && sLowerCaseDeviceType != "acc" && sLowerCaseDeviceType != "nvidia")
        LOG_ERROR("Unkonwn device type " << sDeviceType << ". Acceptable types are 'cpu', 'gpu', 'acc' and 'nvidia'");

    std::vector<cl::Device> const &vAllDevices(GetAllDevices());
    unsigned int uiNumberOfDevices(vAllDevices.size());
    int           iFoundDevice(-1);
    for (unsigned int uiDevice = 0; uiDevice < uiNumberOfDevices; ++uiDevice) {
        if (iFoundDevice != -1)
            break;

        if (vAllDevices[uiDevice].getInfo<CL_DEVICE_VENDOR>() == "NVIDIA" && sLowerCaseDeviceType == "nvidia") // NVIDIA device type
            iFoundDevice = uiDevice;

        if (vAllDevices[uiDevice].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU && sLowerCaseDeviceType == "cpu")
            iFoundDevice = uiDevice;
      
        if (vAllDevices[uiDevice].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU && sLowerCaseDeviceType == "gpu")
            iFoundDevice = uiDevice;
    
        if (vAllDevices[uiDevice].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_ACCELERATOR && sLowerCaseDeviceType == "acc")
            iFoundDevice = uiDevice;
    }

    if (iFoundDevice == -1)
        LOG_ERROR("Unable to find device type " << sDeviceType);
    LOG("Successfully found OpenCL device");
    return vAllDevices[iFoundDevice];
}

OCL::OCL(cl::Device const &SelectedDevice, std::string const &sKernelName, std::string const &sKernelFile)
    : m_clSelectedDevice  (SelectedDevice)
    , m_clContext         (cl::Context({m_clSelectedDevice}))
    , m_clCommandQueue    (cl::CommandQueue(m_clContext, m_clSelectedDevice, CL_QUEUE_PROFILING_ENABLE))
    , m_clProgram         (GetProgram(sKernelFile))
    , m_clKernel          (GetKernel(sKernelName))
    , m_clTotalKernelTime (0)
{
    LOG("Successfully created OpenCL object");
}

/*
    PRIVATE UTILITY FUNCTION
    Returns a string object containing the kernel 

    Input  - sFileName(Path of the kernel file, should exist or errors out)
    Return - String of kernel 
*/
//
std::string OCL::ReadKernelFileFromDiskAsString(std::string const &sFileName) const 
{
    std::ifstream FileStream(sFileName);
    LOG_ASSERT(FileStream, "Error, file " << sFileName << " does not exist!");

    // Slurp in the file and convert it into text
    std::stringstream ss;
    ss << FileStream.rdbuf();
    return ss.str();
}



/*
    PRIVATE UTILITY FUNCTION
    Kernel creation utility, which reads in the .cl file and compiles it during run time. 

    TODO: Should handle multiple kernels

    Input  - sKernelName(Kernel name), sFileName(Path of the kernel file)
    Return - bool 
*/
//
cl::Program OCL::GetProgram(std::string const &sKernelFileName)
{   
	std::error_code _Code;
	std::string const sRealPathOfKernel = std::experimental::filesystem::canonical(sKernelFileName).u8string();
    std::string const sKernelCode      (ReadKernelFileFromDiskAsString(sRealPathOfKernel));
    if (sKernelCode.length() == 0) {
        LOG_ERROR(sKernelFileName << " is empty!");
    }

    LOG("Creating program object from CL kernel file " << sRealPathOfKernel);
    cl::Program::Sources clKernelSources(1, {sKernelCode.c_str(), sKernelCode.size()});
    return cl::Program(m_clContext, clKernelSources);
}

cl::Kernel OCL::GetKernel(std::string const &sKernelName)
{
    Timer tKernelBuild("KernelBuild");
    tKernelBuild.Start();
    cl_uint const uiReturnCode(m_clProgram.build({m_clSelectedDevice}));
    tKernelBuild.Stop();
    LOG_ASSERT(uiReturnCode == CL_SUCCESS, "OpenCL kernel build error. Build Log : " << std::endl << m_clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_clSelectedDevice));

    LOG("Kernel " << sKernelName << " successfully created");
    LOG("\t Using Vendor               : " << m_clSelectedDevice.getInfo<CL_DEVICE_VENDOR>());
    LOG("\t On Device                  : " << m_clSelectedDevice.getInfo<CL_DEVICE_NAME>() << " (Driver Version " << m_clSelectedDevice.getInfo<CL_DRIVER_VERSION>() << ")");
    LOG("\t Max number of Compute Units: " << m_clSelectedDevice.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
    LOG("\t Kernel build time          : " << tKernelBuild.GetTimeAsString(Timer::Units::SECONDS));

    std::vector<std::string> vOpenCLPath(Utility::ExtractLDDPathNameFromProcess({"libOpenCL"}));
    if (vOpenCLPath.empty()) {
        LOG_WARNING("OpenCL path not found! Possible your LD_LIBRARY_PATH is not correctly set");
    } else {
        LOG("\t Using OpenCL Library       : " << (!vOpenCLPath[0].empty() ? vOpenCLPath[0] : " WARNING! OpenCL library not found!"));
    }

    return cl::Kernel(m_clProgram, sKernelName.c_str());
}

/*
    Launch opencl kernel

    TODO: Should handle multiple kernels ....
*/
//
bool OCL::LaunchKernel(cl::NDRange const &offset, cl::NDRange const &globalWorkSize, cl::NDRange const &localWorkSize)
{
    cl::Event clProfilingEvent;
    int const iRetValue(m_clCommandQueue.enqueueNDRangeKernel(m_clKernel, offset, globalWorkSize, localWorkSize, nullptr, &clProfilingEvent));
    LOG_ASSERT(iRetValue == CL_SUCCESS, "Kernel failed to launch, error code : " << iRetValue << " (" << GetErrorMessage(iRetValue) << ")");
   clProfilingEvent.wait();
   m_clTotalKernelTime += GetKernelTime(clProfilingEvent);
    return true;
}

double OCL::GetTotalKernelTime(KernelTimeUnits const &units) const
{
    switch(units) {
        case KernelTimeUnits::SECONDS:      return static_cast<double>(m_clTotalKernelTime) * 1e-9;
        case KernelTimeUnits::MILLISECONDS: return static_cast<double>(m_clTotalKernelTime) * 1e-6;
        case KernelTimeUnits::MICROSECONDS: return static_cast<double>(m_clTotalKernelTime) * 1e-3;
        default: /* NANOSECONDS */          return static_cast<double>(m_clTotalKernelTime);
    }
}


cl_long OCL::GetKernelTime(cl::Event const &clEvent) const
{
    return clEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - clEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
}

std::string OCL::GetErrorMessage(cl_int const clReturnCode) const
{
    switch(clReturnCode){
        // run-time and JIT compiler errors
        case 0:   return "CL_SUCCESS";
        case -1:  return "CL_DEVICE_NOT_FOUND";
        case -2:  return "CL_DEVICE_NOT_AVAILABLE";
        case -3:  return "CL_COMPILER_NOT_AVAILABLE";
        case -4:  return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5:  return "CL_OUT_OF_RESOURCES";
        case -6:  return "CL_OUT_OF_HOST_MEMORY";
        case -7:  return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8:  return "CL_MEM_COPY_OVERLAP";
        case -9:  return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}


