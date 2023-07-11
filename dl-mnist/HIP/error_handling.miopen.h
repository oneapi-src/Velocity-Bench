/* Copyright (C) 2023 Intel Corporation
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 * SPDX-License-Identifier: MIT
 */

#ifndef DEEP_LEARNING_CUDNN_ERROR_HANDLING_LAYER_H_
#define DEEP_LEARNING_CUDNN_ERROR_HANDLING_LAYER_H_

#include <stdio.h>
#include <iostream>
#include <exception>
#include <miopen.h>

using namespace std;


#define assertHipInvar(...)                                                                                    \
    if(int ret_val = __VA_ARGS__) {                                                                                          \
        throw new DlInfraException((hipError_t)ret_val, #__VA_ARGS__, __FILE__, __LINE__);                     \
    }       

#define assertMiopenInvar(...)                                                                                   \
    if(int ret_val = __VA_ARGS__) {                                                                                          \
        throw new DlInfraException((miopenStatus_t)ret_val, #__VA_ARGS__, __FILE__, __LINE__);                   \
    }   

class DlInfraException: public exception
{
    hipError_t hip_func_ret_val_; 
    miopenStatus_t miopen_func_ret_val_;
    const char* call_expression_; 
    const char* filename_; 
    int line_number_;

  public:
    DlInfraException(hipError_t func_ret_val, const char* call_expression, const char* filename, int line_number)
            : hip_func_ret_val_(func_ret_val), miopen_func_ret_val_(miopenStatus_t(0)), call_expression_(call_expression), filename_(filename), line_number_(line_number_) {
                print_exception_msg();
            }

    DlInfraException(miopenStatus_t func_ret_val, const char* call_expression, const char* filename, int line_number) 
        : hip_func_ret_val_((hipError_t)0), miopen_func_ret_val_(func_ret_val), call_expression_(call_expression), filename_(filename), line_number_(line_number_) {
            print_exception_msg();
        }

    void print_exception_msg() {
        //TODO: need to rethink the semantics of miopenStatus_t(0), since it can be confusing
        cout << "Error value returned by call to " << call_expression_  << std::endl;
        if(miopen_func_ret_val_ == miopenStatus_t(0)) {      //0 indicates that no call was made to miopen
            cout << "Error code is " << hip_func_ret_val_ << " (" << hipGetErrorString(hip_func_ret_val_) << ") " << std::endl;
        } else {
            cout << "Error code is " << miopen_func_ret_val_ << " (" << miopenGetErrorString(miopen_func_ret_val_) << ") " << std::endl;
        }
        cout << "At location : " << filename_ << ": " << line_number_ << std::endl;  
         
    } 

    virtual const char* what() const throw()
    {
        std::string errMsg("Error value returned by call to " + std::string(call_expression_) + " at location " + filename_ + ": " + std::to_string(line_number_));
        return errMsg.c_str();
    }
};

#endif