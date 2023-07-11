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

#ifndef CLASS_OCL_H
#define CLASS_OCL_H

#include "Timer.h"

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#include <iostream>
#include <utility>
#include <map>
#include <vector>

class OCL
{
public:
    enum class KernelTimeUnits { SECONDS, MILLISECONDS, MICROSECONDS, NANOSECONDS };
private:
    cl::Device       m_clSelectedDevice;                
    cl::Context      m_clContext;
    cl::CommandQueue m_clCommandQueue;
    cl::Program      m_clProgram;
    cl::Kernel       m_clKernel;
    cl_long          m_clTotalKernelTime;

    // Non object modifying functions
    std::string ReadKernelFileFromDiskAsString(std::string const &sFileName)   const;
    std::string GetErrorMessage               (cl_int      const clReturnCode) const;

    cl_long GetKernelTime(cl::Event const &clEvent) const; // There is no unit conversion, developer is responsible

    cl::Program  GetProgram (std::string  const &sKernelFile);
    cl::Kernel   GetKernel  (std::string  const &sKernelName);

    static std::vector<cl::Device> GetAllDevices(); 
public:
    OCL(cl::Device  const &SelectedDevice, std::string const &sKernelName, std::string const &sKernelFile);
   ~OCL(){}
    
    // Disable copying
    OCL           (OCL const &rightOCL) = delete; 
    OCL &operator=(OCL const &rightOCL) = delete;

    bool LaunchKernel(cl::NDRange const &offset, cl::NDRange const &globalWorkSize, cl::NDRange const &localWorkSize); //TODO: Support multiple kernels?

    cl::Context      &GetCLContext()       { return m_clContext;     } // This is needed to create OpenCL buffers
    cl::CommandQueue &GetCommandQueue()    { return m_clCommandQueue;}

    double GetTotalKernelTime() const      { return GetTotalKernelTime(OCL::KernelTimeUnits::SECONDS); }
    double GetTotalKernelTime(KernelTimeUnits const &units) const;

    static  cl::Device GetDevice(std::string const &sDeviceType);
public: // Templated functions
    template<class T>
    void SetArgument(int const iIndex, T ArgumentObject)           
    {
        int const iRetValue(m_clKernel.setArg(iIndex, ArgumentObject));
        if (iRetValue != CL_SUCCESS) { 
            std::cerr << "OCLC ERROR: Argument #" << iIndex << " could not be set. Error code : " << iRetValue << " (" << GetErrorMessage(iRetValue) << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    template<class T>
    void EnqueueWriteBuffer(cl::Buffer &BufferObject, size_t const uiBytesToWrite, T *pData)
    {
        int const iRetValue(m_clCommandQueue.enqueueWriteBuffer(BufferObject, CL_TRUE, 0, uiBytesToWrite, pData));
        if (iRetValue != CL_SUCCESS) {
            std::cerr << "OCLC ERROR: enqueueWriteBuffer cannot be completed. Error code : " << iRetValue << " (" << GetErrorMessage(iRetValue) << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    template<class T>
    void EnqueueReadBuffer(cl::Buffer &BufferObject, size_t const uiBytesToRead, T *pData)
    {
        int const iRetValue(m_clCommandQueue.enqueueReadBuffer(BufferObject, CL_TRUE, 0, uiBytesToRead, pData));
        if (iRetValue != CL_SUCCESS) {
            std::cerr << "OCLC ERROR: enqueueReadBuffer cannot be completed. Error code : " << iRetValue << " (" << GetErrorMessage(iRetValue) << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
};

#endif
