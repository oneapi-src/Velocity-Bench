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

#include <chrono>
#include "utils.h"
#include "tracer.h"

using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
using namespace dl_infra::common;

// // Generate uniform numbers [0,1)
void initImage(float* image, int imageSize) {
    Tracer::func_begin("initImage");           

    unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = float(seed) * 2.3283064e-10;  // 2^-32
    }

    Tracer::func_end("initImage");                 
}


double get_total_time() {
    Tracer::func_begin("get_total_time");   

    double sum = 0;
    for(int i=0; i<tl.size(); i++) {
        auto tuple = tl[i];
        sum += get<1>(tuple);
        cout << get<0>(tuple) << ", " << get<1>(tuple) << std::endl;
    }

    Tracer::func_end("get_total_time");    

    return sum;
}