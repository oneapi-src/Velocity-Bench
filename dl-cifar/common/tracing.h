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

#ifndef DL_CIFAR_TRACER_H_
#define DL_CIFAR_TRACER_H_

#include <iostream>
#include <string>


namespace dl_cifar::common {
    class Tracer {
        private:
            static bool trace_func_boundaries, trace_conv_op, trace_mem_op;

        public:
            static void set_trace_func_boundaries() {trace_func_boundaries = true;}
            static void unset_trace_func_boundaries() {trace_func_boundaries = false;}
            static void set_trace_conv_op() {trace_conv_op = true;}
            static void unset_trace_conv_op() {trace_conv_op = false;}
            static void set_trace_mem_op() {trace_mem_op = true;}
            static void unset_trace_mem_op() {trace_mem_op = false;}

            static void func_begin(std::string func_name) 
            {
                if(trace_func_boundaries) {
                        std::cout<< "BEGIN FUNC - " << func_name << std::endl;
                }
            }

            static void func_end(std::string func_name) 
            {
                if(trace_func_boundaries) {
                        std::cout<< "END FUNC   - " << func_name << std::endl;
                }
            }

            static void conv_op(std::string trace_log) 
            {
                if(trace_conv_op) {
                        std::cout<< trace_log << std::endl;
                }
            }

            static void mem_op(std::string trace_log) 
            {
                if(trace_mem_op) {
                        std::cout<< trace_log << std::endl;
                }
            }
    };
};
#endif