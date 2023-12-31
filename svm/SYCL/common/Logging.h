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

#include <iostream>
#include <sstream>

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

#ifdef ENABLE_CUDA_LOGGING
#include <cuda.h>
                                                                                                                                                                                                                                                                                          
#define checkCUDA(expression)                               \
  {                                                          \
    cudaError_t const status(expression);                     \
    if (status != cudaSuccess) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudaGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }                                

#endif
