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

#include "timing.h"
#include "utils.h"
#include "tracer.h"
#include <chrono>
#include <bits/stdc++.h>
#include <string>



using namespace std;
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
using namespace dl_infra::common;

namespace dl_infra {
    namespace common {
        void Timer::recordOpTimeTaken(int layerIndex, double elapsedTime, string opTag) 
        {
            //std::cout << "Adding " << opTag << ", " << elapsedTime << std::endl;
            tl_.push_back( tuple<string, double>(opTag, elapsedTime) );
            total_ops_time_taken_ += elapsedTime;
            //std::cout << "total_ops_time_taken_ = " << total_ops_time_taken_ << std::endl; 
            //std::cout << "opTag vector size = " << tl_.size() << std::endl;

            Tracer::conv_op("TIMER -  exec_time: " + to_string(elapsedTime) + ", \t\ttotal_ops_time: " + to_string(total_ops_time_taken_) + ", OP_list_count: " + to_string(tl_.size()) + ", \t\tLayer: " + to_string(layerIndex) + ", OP: " + opTag);
        }

        void Timer::reset() {
            total_ops_time_taken_ = 0.0;
        }
    }
}