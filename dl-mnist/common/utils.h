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

#ifndef DEEP_LEARNING_UTILS_H_
#define DEEP_LEARNING_UTILS_H_

#include <chrono>
#include <bits/stdc++.h>


using namespace std;
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

typedef vector< tuple<string, double> > my_tuple;
static my_tuple tl; 
static double timing_sum = 0; 

inline Time get_time_now() 
{
    return std::chrono::high_resolution_clock::now();
}

inline double get_sum() 
{
    return timing_sum;
}

inline double calculate_op_time_taken(Time start) 
{
    Time end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> d = end - start;
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(d);
    double elapsedTime = us.count() / 1000.0f;
    return elapsedTime / 1000.0f;    // in sec
}


// Experimental
inline void capture_timing_func() {

}

// Experimental
#define capture_timing(...)                                                        \
        Time start = get_time_now();                                               \
        int err = __VA_ARGS__;                                                     \
        double elTime = calculate_op_time_taken(start) / 1000.0f;                     \
        totTime += elTime;                                                   \
        if (err) {                                                                 \
            cout << "hi" << std::endl;                                             \
            capture_timing_func();                                                 \
        }                                                                          \



// // Generate uniform numbers [0,1)
void initImage(float* image, int imageSize);
double get_total_time();


#endif