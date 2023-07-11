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
#include <cstring>
#include <exception>
#include <cudnn.h>

using namespace std;


#define assertCudaInvar(...)                                                                                    \
    if(int ret_val = __VA_ARGS__) {                                                                                          \
        throw DlInfraException((cudaError_t)ret_val, #__VA_ARGS__, __FILE__, __LINE__);                     \
    }       

#define assertCudnnInvar(...)                                                                                   \
    if(int ret_val = __VA_ARGS__) {                                                                                          \
        throw DlInfraException((cudnnStatus_t)ret_val, #__VA_ARGS__, __FILE__, __LINE__);                   \
    }   

class DlInfraException: public exception
{
    cudaError_t cuda_func_ret_val_; 
    cudnnStatus_t cudnn_func_ret_val_;
    const char* call_expression_; 
    const char* filename_; 
    int line_number_;

  public:
    DlInfraException(cudaError_t func_ret_val, const char* call_expression, const char* filename, int line_number)
            : cuda_func_ret_val_(func_ret_val), cudnn_func_ret_val_(cudnnStatus_t(0)), call_expression_(call_expression), filename_(filename), line_number_(line_number) {
                print_exception_msg();
            }

    DlInfraException(cudnnStatus_t func_ret_val, const char* call_expression, const char* filename, int line_number) 
        : cuda_func_ret_val_((cudaError_t)0), cudnn_func_ret_val_(func_ret_val), call_expression_(call_expression), filename_(filename), line_number_(line_number) {
            print_exception_msg();
        }

    void print_exception_msg() {
        //TODO: need to rethink the semantics of cudnnStatus_t(0), since it can be confusing
        cout << "Error value returned by call to " << call_expression_  << std::endl;
        if(cudnn_func_ret_val_ == cudnnStatus_t(0)) {      //0 indicates that no call was made to CUDNN
            cout << "Error code is " << cuda_func_ret_val_ << " (" << cudaGetErrorString(cuda_func_ret_val_) << ") " << std::endl;
        } else {
            cout << "Error code is " << cudnn_func_ret_val_ << " (" << cudnnGetErrorString(cudnn_func_ret_val_) << ") " << std::endl;
        }
        cout << "At location : " << filename_ << ": " << line_number_ << std::endl;  
         
    } 

    virtual const char* what() const throw()
    {
        std::string errMsg("Error value returned by call to " + std::string(call_expression_) + " at location " + filename_ + ": " + std::to_string(line_number_));
        char* char_array = new char[errMsg.length() + 1];
        strcpy(char_array, errMsg.c_str());
        return char_array;
    }
};

#endif