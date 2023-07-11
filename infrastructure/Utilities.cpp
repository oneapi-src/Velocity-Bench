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


#include "Utilities.h"
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <set>
#include <algorithm>

#ifdef WIN64
#include <direct.h>
#else
#include <unistd.h>
#endif

#define CPP_MODULE "UTIL"
#include "Logging.h"

/*
    Transform upper case text

    Input  - sInput(String to conver to uppercase)
    Return - String in uppercase
 */
//
//
std::string Utility::ToUpperCase(std::string const &sInput)
{
    std::string s(sInput);
#ifdef WIN64
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) -> unsigned char { return      toupper(c); });
#else
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) -> unsigned char { return std::toupper(c); }); 
#endif
    return s;
}

std::string Utility::ToLowerCase(std::string const &sInput)
{
    std::string s(sInput);
#ifdef WIN64
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) -> unsigned char { return      tolower(c); });
#else
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) -> unsigned char { return std::tolower(c); }); 
#endif
    return s;
}

/*
    Used for obtaining the correct location of the CUDA / OpenCL kernel 
 */
//
//
std::string Utility::GetCurrentRunningDirectory()
{
    char temp[4096];
#ifdef WIN64
    return (_getcwd(temp, 4096) ? std::string( temp ) : std::string(""));
#else
    return (getcwd(temp, 4096) ? std::string( temp ) : std::string(""));
#endif
}

std::string Utility::GetHostName()
{
#ifdef WIN64
	return "Windows";
#else
    char cHostName[4096];
    gethostname(cHostName, 4096);
    return std::string(cHostName);
#endif
}

std::string Utility::GetProcessID()
{
#ifdef WIN64
    return "N/A";
#else
    return std::to_string(::getpid());
#endif
}

std::vector<std::string> Utility::ExtractLDDPathNameFromProcess(std::vector<std::string> const &vLDDPathsToSearch)
{
    std::string const sProcessMapsFile("/proc/" + GetProcessID() + "/maps");
    std::ifstream inMapsFile(sProcessMapsFile);
    if (!inMapsFile.good()) { 
        LOG_WARNING("Unable to find process's maps file " << sProcessMapsFile);
        return std::vector<std::string>(); // Return empty vector
    }

    std::set<std::string> setUniquePathsFound;   
    while (!inMapsFile.eof()) {
        std::string sStringLine("");
        std::getline(inMapsFile, sStringLine);
        if (sStringLine.find_first_of('/') == std::string::npos)
            continue;
        setUniquePathsFound.insert(sStringLine.substr(sStringLine.find_first_of('/'), sStringLine.length()));
    }
 
    unsigned int const uiNumberOfPathsToSearch(vLDDPathsToSearch.size());
    std::vector<std::string> vFoundLDDPaths(uiNumberOfPathsToSearch, "");
    for (unsigned int uiPath = 0; uiPath < uiNumberOfPathsToSearch; ++uiPath) {
        for (auto const &sPath : setUniquePathsFound) {
            if (sPath.find(vLDDPathsToSearch[uiPath]) == std::string::npos)
                continue;
            vFoundLDDPaths[uiPath] = sPath;
        }
    }

    return vFoundLDDPaths;
}

float Utility::RandomFloatNumber(double const dBeginRange, double const dEndRange)
{                                                                                                                                                                                                                  
    std::random_device rd;
    std::mt19937_64 mt{rd()};
    std::uniform_real_distribution<double> uniformDouble(dBeginRange, dEndRange);
    return uniformDouble(mt);
}

int Utility::RandomIntegerNumber(int const iBeginRange, int const iEndRange)
{                                                                                                                                                                                                                  
    std::random_device rd;
    std::mt19937_64 mt{rd()};
    std::uniform_int_distribution<int> uniformDouble(iBeginRange, iEndRange);
    return uniformDouble(mt);
}

#ifndef WIN64
std::string Utility::GetEnvironmentValue(std::string const &sEnvironmentVariableName)
{
    char const *pEnvironmentValue(::getenv(sEnvironmentVariableName.c_str()));
    if (pEnvironmentValue == nullptr)
        return "<not set>";
    return pEnvironmentValue;
}
#endif

std::string Utility::DisplayHumanReadableBytes(size_t const uBytes)
{
    std::vector<std::string> const asUnits({"B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"});
    int iUnitPosition(0);
    size_t uConvertedBytes(uBytes);
    while (uConvertedBytes > 1024) {
        uConvertedBytes /= 1024;
        ++iUnitPosition;
    }
    return std::to_string(uConvertedBytes) + asUnits[iUnitPosition];
}

std::string Utility::ConvertTimeToReadable(double const tTimeInNanoSeconds)
{
    std::vector<std::string> const asTimeUnits({"ns", "us", "ms", "s"});
    int iTimeUnits(0);
    double Time(tTimeInNanoSeconds);
    while(std::isgreater(Time, 1000.0) && iTimeUnits < 4) {
        Time /= 1000.0;
        ++iTimeUnits;
    }
    return std::to_string(Time) + " " +  asTimeUnits[iTimeUnits];
}

bool Utility::IsHTEnabled()
{
    std::string const sThreadSiblingsFile("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list"); // If a CPU exists, then CPU0 should exist
    std::ifstream inThreadSiblingsFile(sThreadSiblingsFile);
    if (!inThreadSiblingsFile.good()) {
        LOG_WARNING("Cannot determine if CPU has HT enabled");
        return false;
    }

    std::string sStringLine("");
    std::getline(inThreadSiblingsFile, sStringLine);
    if (sStringLine.find(",") != std::string::npos)
        return true;
    return false;
}

bool Utility::IsHTEnabled(std::string const &sParentCPUName)
{
    std::string const &sUpperCaseCPUName(ToUpperCase(sParentCPUName));
    return sUpperCaseCPUName.find("XEON") != std::string::npos;
}

bool Utility::IsInteger(std::string const &sStringToCheck)
{
#ifdef WIN64
    return std::find_if(sStringToCheck.begin() + (sStringToCheck.front() == '-' ? 1 : 0), sStringToCheck.end(), [](char c) { return !isdigit(c); }) == sStringToCheck.end();
#else
    return std::find_if(sStringToCheck.begin() + (sStringToCheck.front() == '-' ? 1 : 0), sStringToCheck.end(), [](char c) { return !std::isdigit(c); }) == sStringToCheck.end();
#endif
}

std::vector<std::string> Utility::TokenizeString(std::string const &sStringToTokenize, char const delim)
{
    std::stringstream ss(sStringToTokenize);
     std::vector<std::string> vTokens;
     while (!ss.eof()) {
         std::string sToken;
         std::getline(ss, sToken, delim);
         sToken.erase(std::remove_if(sToken.begin(), sToken.end(), ::isspace), sToken.end());
         vTokens.push_back(std::move(sToken));
     }
     return vTokens;
}

#ifdef USE_CUDA

#undef CPP_MODULE
#define CPP_MODULE "CUDA" // This might overwrite Utilities CPP_MODULE name from UTIL
                          // May consider moving these functions into a CUDA helper file

void Utility::PrintCUDADeviceProperty(const cudaDeviceProp& prop) {
    LOG("\t Device name                 : " << prop.name);
    LOG("\t Memory Clock Rate (GHz)     : " << prop.memoryClockRate * 1e-6);
    LOG("\t Memory Bus Width (bits)     : " << prop.memoryBusWidth);
    LOG("\t CUDA cores                  : " << prop.multiProcessorCount);
    LOG("\t Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    LOG("\t Compute capability          : " << prop.major << "." << prop.minor);
}
void Utility::QueryCUDADevice()
{
    LOG("Querying CUDA devices on " << Utility::GetHostName());
    int iNumDevices(-1);
    checkCUDA(cudaGetDeviceCount(&iNumDevices));
    if (iNumDevices <= 0) 
        LOG_ERROR(Utility::GetHostName() << " does not have any available NVIDIA devices");
    LOG("Number of CUDA devices found: " << iNumDevices);
    for(int i=0; i<iNumDevices; i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        LOG("*** Device Number: " << i << " *** ");
        PrintCUDADeviceProperty(prop);
    }
    checkCUDA(cudaSetDevice(0)); // Use the first NVIDIA device
}
#elif USE_HIP
#undef CPP_MODULE
#define CPP_MODULE "HIPC" // This might overwrite Utilities CPP_MODULE name from UTIL
                          // May consider moving these functions into a CUDA helper file

void Utility::PrintHIPDeviceProperty(hipDeviceProp_t const &prop) {
    LOG("\t Device name                 : " << prop.name);
    LOG("\t Memory Clock Rate (GHz)     : " << prop.memoryClockRate * 1e-6);
    LOG("\t Memory Bus Width (bits)     : " << prop.memoryBusWidth);
    LOG("\t Compute cores               : " << prop.multiProcessorCount);
    LOG("\t Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    LOG("\t Compute capability          : " << prop.major << "." << prop.minor);
}
void Utility::QueryHIPDevice()
{
    LOG("Querying HIP devices on " << Utility::GetHostName());
    int iNumDevices(-1);
    checkHIP(hipGetDeviceCount(&iNumDevices));
    if (iNumDevices <= 0) 
        LOG_ERROR(Utility::GetHostName() << " does not have any available NVIDIA devices");
    LOG("Number of HIP devices found: " << iNumDevices);
    for(int i=0; i<iNumDevices; i++){
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        LOG("*** Device Number: " << i << " *** ");
        PrintHIPDeviceProperty(prop);
    }
    checkHIP(hipSetDevice(0)); // Use first HIP device
}

#endif
