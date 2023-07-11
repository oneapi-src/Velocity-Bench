/*
 * Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU Lesser General Public License v3.0 or later
 * 
 * If a copy of the license was not distributed with this file, you can obtain one at 
 * https://www.gnu.org/licenses/lgpl-3.0-standalone.html
 * 
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


/*
    Logging utilities

    USAGE: create a macro called CPP_MODULE, followed by including this file

    e.g.,

    #define CPP_MODULE "MAIN"
    #include "Logging.h" <<< This will not redefine CPP_MODULE
*/
//
//

#include <iostream>
#include <sstream>
#include <cassert>

#ifndef CPP_MODULE
#define CPP_MODULE "UNKN"
#endif

// General logging message
#define LOG(msg) std::cout << CPP_MODULE << ": " << msg << "\n";

// Warning logging message
#define LOG_WARNING(msg) std::cout << CPP_MODULE << " WARNING (" << __LINE__ << "): " << msg << "\n";

// Logging with error, throws an exception as well
#define LOG_ERROR(msg) {\
    std::stringstream sErrorMessage;\
    sErrorMessage << CPP_MODULE << " ERROR(" << __LINE__<< "): " <<  msg << "\n"; \
    std::cerr << sErrorMessage.str(); \
    throw std::runtime_error(sErrorMessage.str()); }\

// Assertion check
#define LOG_ASSERT(cond, msg)\
    do { \
        if (!(cond)) { \
            std::cerr << CPP_MODULE << " ERROR(" << __LINE__<< "): " <<  msg << "\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while(0) \

//#endif

#ifdef USING_CUDA
#include <cuda.h>

#define checkCUDA(expression)                               \
  {                                                          \
    cudaError_t const status(expression);                     \
    if (status != cudaSuccess) {                    \
      std::stringstream sErrorMessage; \
      sErrorMessage << "Error in  " << __FILE__ << ":" << __LINE__ << " "      \
                << cudaGetErrorString(status) << "\n"; \
      throw std::runtime_error(sErrorMessage.str()); \
  }\
  }\

#define checkLastCUDAError()                               \
  {                                                          \
    cudaError_t const status(cudaGetLastError());                     \
    if (status != cudaSuccess) {                    \
      std::stringstream sErrorMessage; \
      sErrorMessage << "Error in  " << __FILE__ << ":" << __LINE__ << " "      \
                << cudaGetErrorString(status) << "\n"; \
      throw std::runtime_error(sErrorMessage.str()); \
  }\
  }\

#endif

#ifdef ENABLE_HIP_LOGGING
#include <hip/hip_runtime.h>

#define checkHIP(expression)                               \
  {                                                          \
    hipError_t const status(expression);                     \
    if (status != hipSuccess) {                    \
      std::stringstream sErrorMessage; \
      sErrorMessage << "Error on line " << __LINE__ << ": "      \
                << hipGetErrorString(status) << "\n"; \
      throw std::runtime_error(sErrorMessage.str()); \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }                                                          \


#define checkLastHIPError()                               \
  {                                                          \
    hipError_t const status(hipGetLastError());                     \
    if (status != hipSuccess) {                    \
      std::stringstream sErrorMessage; \
      sErrorMessage << "Error in  " << __FILE__ << ":" << __LINE__ << " "      \
                << hipGetErrorString(status) << "\n"; \
      throw std::runtime_error(sErrorMessage.str()); \
  }\
  }\



#endif



