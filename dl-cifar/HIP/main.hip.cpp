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

#include <iostream>
#include <string>

#include "vit/vit.h"
#include "cait/cait.h"

#include "timing.h"
#include "Utilities.h"
#include "workload_params.h"
#include <chrono>
#include "handle.h"
#include "upsample.h"

using namespace dl_cifar;
using namespace dl_cifar::common;


int main(int argc, const char** argv) {
    Timer* timer = new Timer();
    Timer* dataFileReadTimer = new Timer();

    Time wallClockStart = get_time_now();
    Time start;

    try {
        std::cout << std::endl << "\t\tWelcome to DL-CIFAR workload: HIP version." << std::endl << std::endl;
        std::cout << "=======================================================================" << std::endl;
        
        LangHandle *langHandle = new LangHandle(timer);

        Utility::QueryHIPDevice();
        std::cout << "=======================================================================" << std::endl;
        std::cout << std::endl;

        DlCifarWorkloadParams workload_params(argc, argv);

        Time wallClockCompute = get_time_now();
        VitController::execute(workload_params.getDlNwSizeType(), langHandle, timer, dataFileReadTimer);
        CaitController::execute(workload_params.getDlNwSizeType(), langHandle, timer, dataFileReadTimer);

        double computeTime = calculate_op_time_taken(wallClockCompute);
        double totalTime = calculate_op_time_taken(wallClockStart);
        double ioTime = dataFileReadTimer->getTotalOpTime();
               
        std::cout << "dl-cifar - I/O time: " << ioTime << " s" << std::endl;
        std::cout << "dl-cifar - compute time: " << computeTime - ioTime << " s" << std::endl;
        std::cout << "dl-cifar - total time for whole calculation: " << totalTime - ioTime << " s" << std::endl;

        delete timer;
        delete dataFileReadTimer;
    } catch (std::runtime_error& e) {
        std::cout << e.what() << "\n";
        std::cout << "Workload execution failed" << std::endl;
        return -1;
    }

    //hipDeviceReset();
    
}