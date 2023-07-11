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

#ifndef DEEP_LEARNING_TRACER_H_
#define DEEP_LEARNING_TRACER_H_

#include <iostream>
#include <string>

using namespace std;

namespace dl_infra {
    namespace common {
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

                static void func_begin(string func_name) 
                {
                    if(trace_func_boundaries) {
                            cout<< "BEGIN FUNC - " << func_name << endl;
                    }
                }

                static void func_end(string func_name) 
                {
                    if(trace_func_boundaries) {
                            cout<< "END FUNC   - " << func_name << endl;
                    }
                }

                static void conv_op(string trace_log) 
                {
                    if(trace_conv_op) {
                            cout<< trace_log << endl;
                    }
                }

                static void mem_op(string trace_log) 
                {
                    if(trace_mem_op) {
                            cout<< trace_log << endl;
                    }
                }

                

        };
    }
}
#endif