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

#ifndef DL_CIFAR_TIMING_H_
#define DL_CIFAR_TIMING_H_

#include <string>
#include <vector>
#include <bits/stdc++.h>
#include <chrono>
#include "tracing.h"

using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;


namespace dl_cifar::common {
    class Timer {
        private:
            typedef std::vector< std::tuple<std::string, double> > my_tuple_;
            my_tuple_ tl_; 
            double total_ops_time_taken_ = 0.0;

        public:
            void recordOpTimeTaken(int layerIndex, double elapsedTime, std::string opTag); 
            double getTotalOpTime() { return total_ops_time_taken_;};
            void reset();
    };

    inline Time get_time_now() {
        return std::chrono::high_resolution_clock::now();
    }
    inline double calculate_op_time_taken(Time start) {
        return (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.0f)/1000.0f;
    }


};

#endif
