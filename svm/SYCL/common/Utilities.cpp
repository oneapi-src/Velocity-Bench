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


#include "Utilities.h"
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <set>
#include<cmath>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
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
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) -> unsigned char { return      toupper(c); });
#else
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) -> unsigned char { return std::toupper(c); }); 
#endif
    return s;
}

std::string Utility::ToLowerCase(std::string const &sInput)
{
    std::string s(sInput);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
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
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    return (_getcwd(temp, 4096) ? std::string( temp ) : std::string(""));
#else
    return (getcwd(temp, 4096) ? std::string( temp ) : std::string(""));
#endif
}

std::string Utility::GetHostName()
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	return "Windows";
#else
    char cHostName[4096];
    gethostname(cHostName, 4096);
    return std::string(cHostName);
#endif
}

bool Utility::FileExists(std::string const &sFileName)
{
    std::ifstream ifFileChecker(sFileName.c_str(), std::ios::in);
    if (!ifFileChecker.good()) {
        LOG_ERROR(sFileName << " was not found or inaccessible");
        return false;
    }
    return true; 
}

std::string Utility::GetProcessID()
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
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

std::string Utility::ConvertTimeToReadable(long long const tTimeInNanoSeconds)
{
    std::vector<std::string> const asTimeUnits({"ns", "us", "ms", "s"});
    int iTimeUnits(0);
    double Time(tTimeInNanoSeconds);
    // Unify time to seconds
    while(/*std::isgreater(Time, 1000.0) && */iTimeUnits < 3) {
        Time /= 1000.0;
        ++iTimeUnits;
    }
    return std::to_string(Time) + asTimeUnits[iTimeUnits];
}

#ifdef USE_CUDA
void Utility::QueryCUDADevice()
{
    int iNumDevices(-1);
    checkCUDA(cudaGetDeviceCount(&iNumDevices));
    LOG("Number of CUDA devices found: " << iNumDevices);
    if (iNumDevices < 0) 
        LOG_ERROR(Utility::GetHostName() << " does not have any available NVIDIA devices");
}
#endif
