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

#include<hip/hip_runtime.h>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>

#include "common.hpp"
#include "Utilities.h"
#include "Logging.h"

#define TIMER_START() time_start = std::chrono::steady_clock::now();
#define TIMER_END()                                                                         \
    time_end = std::chrono::steady_clock::now();                                            \
    time_total  = std::chrono::duration<double, std::milli>(time_end - time_start).count();
#define TIMER_PRINT(name) std::cout << name <<": " << (time_total - time_total_) / 1e3 << " s\n";

#define TIMER_START_() time_start_ = std::chrono::steady_clock::now();
#define TIMER_END_()                                                                         \
    time_end_ = std::chrono::steady_clock::now();                                            \
    time_total_  += std::chrono::duration<double, std::milli>(time_end_ - time_start_).count();
#define TIMER_PRINT_(name) std::cout << name <<": " << time_total_ / 1e3 << " s\n";

#define START_TIMER() start_time = std::chrono::steady_clock::now();
#define STOP_TIMER()                                                                        \
    stop_time = std::chrono::steady_clock::now();                                           \
    duration  = std::chrono::duration<double, std::milli>(stop_time - start_time).count();  \
    tot_time += duration;
#define PRINT_TIMER(name) std::cout <<name <<"      :" << duration << " ms\n";

// #ifndef DEBUG_TIME
// #define DEBUG_TIME
// #endif

#undef CPP_MODULE
#define CPP_MODULE "HIMN"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

using namespace cv;
using namespace std;

#define HIP_CHECK(call) {                                                   \
    hipError_t err = call;                                                  \
    if( hipSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
            __FILE__, __LINE__, hipGetErrorString( err) );                  \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

#define HIP_CHECK_LAST_ERROR() {                                            \
    if( hipSuccess != hipGetLastError()) {                                  \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
            __FILE__, __LINE__, hipGetErrorString( hipGetLastError() ) );   \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

__global__
void computeGradient(
    unsigned char* input,
    unsigned char* output,
    int rows,
    int cols)
{
    float gradientx[3][3] = {{-1.f,  0.f,  1.f}, {-2.f, 0.f, 2.f}, {-1.f, 0.f, 1.f}}; 
    float gradienty[3][3] = {{-1.f, -2.f, -1.f}, { 0.f, 0.f, 0.f}, { 1.f, 2.f, 1.f}};

    float gradient_x, gradient_y;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows - 2 || col >= cols - 2) return;

    int index = row * cols + col + cols + 1;
    int index_row_above = index - cols;
    int index_row_below = index + cols;

    gradient_x =    (gradientx[0][0] * input[index_row_above - 1])    +
                    (gradientx[0][1] * input[index_row_above])        +
                    (gradientx[0][2] * input[index_row_above + 1])    +
                    (gradientx[1][0] * input[index - 1])              +
                    (gradientx[1][1] * input[index])                  +
                    (gradientx[1][2] * input[index + 1])              +
                    (gradientx[2][0] * input[index_row_below - 1])    +
                    (gradientx[2][1] * input[index_row_below])        +
                    (gradientx[2][2] * input[index_row_below + 1]);

    gradient_y =    (gradienty[0][0] * input[index_row_above - 1])    +
                    (gradienty[0][1] * input[index_row_above])        +
                    (gradienty[0][2] * input[index_row_above + 1])    +
                    (gradienty[1][0] * input[index - 1])              +
                    (gradienty[1][1] * input[index])                  +
                    (gradienty[1][2] * input[index + 1])              +
                    (gradienty[2][0] * input[index_row_below - 1])    +
                    (gradienty[2][1] * input[index_row_below])        +
                    (gradienty[2][2] * input[index_row_below + 1]);

    output[index] = sqrtf(powf(gradient_x, 2.f) + powf(gradient_y, 2.f));
}

int main(int argc, const char* argv[])
{
    std::chrono::steady_clock::time_point time_start;
    std::chrono::steady_clock::time_point time_end;
    double time_total = 0.0;

    std::chrono::steady_clock::time_point time_start_;
    std::chrono::steady_clock::time_point time_end_;
    double time_total_ = 0.0;

    TIMER_START()

    try {
    LOG("Welcome to the HIP version of Sobel filter workload.");

    VelocityBench::CommandLineParser parser;
    InitializeCmdLineParser(parser);
    parser.Parse(argc, argv);

    // int iScaleFactor(parser.GetIntegerSetting("-f"));
    std::string inputfile = parser.GetSetting("-i");
    int nIterations = parser.GetIntegerSetting("-n");
    if(nIterations < 0 || nIterations > 100) {
        LOG_ERROR("# of iterations must be within range [1, 100]");
    }

    // TIMER_START_()
    // LOG("Input image file: "<< inputfile);
    // Mat inputimage = imread(inputfile, IMREAD_COLOR);
    // if (inputimage.empty()) {
    //     LOG_ERROR("Failed to open input image\n");
    //     return -1;
    // }
    // TIMER_END_()

    TIMER_START_()
    LOG("Input image file: "<< inputfile);
    Mat scaledImage = imread(inputfile, IMREAD_GRAYSCALE);
    if (scaledImage.empty()) {
        LOG_ERROR("Failed to open input image\n");
        return -1;
    }
    TIMER_END_()

    // LOG("Scaling image...");
    // Mat scaledImage = preprocess(inputimage, iScaleFactor);
    // LOG("Scaling image...done");
    // imwrite("../../res/silverfalls_32Kx32K.png", scaledImage);
    // std::cout << "\nsizes: " << inputimage.rows << " " << inputimage.cols << " " << scaledImage.rows << " "  << scaledImage.cols << "\n\n";

    int rows = scaledImage.rows;
    int cols = scaledImage.cols;
    size_t globalsize = rows * cols;
    Mat gradientimage(rows, cols, CV_8UC1);

    LOG("Launching HIP kernel with # of iterations: "<< nIterations);

    int counter(nIterations);

#ifdef DEBUG_TIME
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point stop_time;
    double duration = 0.0;
    double tot_time = 0.0;

    START_TIMER();
#endif

    HIP_CHECK( hipSetDevice(0) );

#ifdef DEBUG_TIME
    STOP_TIMER();
    PRINT_TIMER("init     ");
    std::cout << std::endl;
#endif

    while(counter > 0) {

#ifdef DEBUG_TIME
        START_TIMER();
#endif

        //Allocate the device memory
        unsigned char *d_input, *d_gradient;
        HIP_CHECK( hipMalloc(&d_input,    globalsize * sizeof(unsigned char)) );
        HIP_CHECK( hipMalloc(&d_gradient, globalsize * sizeof(unsigned char)) );
#ifdef DEBUG_TIME
        STOP_TIMER();
        std::cout << "Iteration: " << counter << std::endl;
        PRINT_TIMER("malloc   ");

        START_TIMER();
#endif

        //Copy source vectors from host to device
        HIP_CHECK( hipMemcpy(d_input, scaledImage.data, globalsize * sizeof(unsigned char), hipMemcpyHostToDevice) );
        HIP_CHECK( hipDeviceSynchronize() );

#ifdef DEBUG_TIME
        STOP_TIMER();
        PRINT_TIMER("memcpyH2D");

        START_TIMER();
#endif

        //Step 3 Gradient strength and direction
        constexpr int blockDim_x = 64;
        constexpr int blockDim_y = 2;
        dim3 block(blockDim_x, blockDim_y);
        dim3 grid((cols - 2 + blockDim_x - 1) / blockDim_x, (rows - 2 + blockDim_y - 1) / blockDim_y);
        hipLaunchKernelGGL(computeGradient, grid, block, 0, 0, d_input, d_gradient, rows, cols);
        HIP_CHECK( hipDeviceSynchronize() );

#ifdef DEBUG_TIME
        STOP_TIMER();
        PRINT_TIMER("kernel   ");

        START_TIMER();
#endif

        //Copy back result from device to host
        HIP_CHECK( hipMemcpy(gradientimage.data, d_gradient, globalsize * sizeof(unsigned char), hipMemcpyDeviceToHost) );
        HIP_CHECK( hipDeviceSynchronize() );

#ifdef DEBUG_TIME
        STOP_TIMER();
        PRINT_TIMER("memcpyD2H");
#endif

        //Free up device allocation
        HIP_CHECK( hipFree(d_input) );
        HIP_CHECK( hipFree(d_gradient) );
        counter--;

#ifdef DEBUG_TIME
        std::cout << std::endl;
#endif
    }

    TIMER_START_()
    Mat outputimage = cv::Mat::zeros(rows - 2, cols - 2, CV_8UC1);
    for(int i = 0; i< rows - 2; i++) {
        for(int j = 0; j< cols - 2; j++) {
            outputimage.at<uchar>(i, j) = gradientimage.at<uchar>(i + 1, j + 1);
        }
    }

    // write output
    if(parser.IsSet("-o")) {
        std::string outputfile = parser.GetSetting("-o");
        LOG("Writing output image into: "<< outputfile);
        if(outputfile.empty()) {
            LOG_ERROR("Invalid output filename provided.");
        }
        imwrite(outputfile, outputimage);
    }

    // run verification
    if(parser.IsSet("-v")) {
        Mat refimage = compute_reference_image(scaledImage);
        verify_results(outputimage, refimage, 5);
        if(parser.IsSet("-saveref")) {
            LOG("Saving reference image for debugging.");
            imwrite("./scalar.bmp", refimage);
        }
    }
    } catch (std::exception const& e) {
        std::cout << "Exception: " << e.what() << "\n";
    }
    TIMER_END_()
    TIMER_PRINT_("time to subtract from total")

    TIMER_END()
    TIMER_PRINT("sobelfilter - total time for whole calculation")

    return 0;
}
