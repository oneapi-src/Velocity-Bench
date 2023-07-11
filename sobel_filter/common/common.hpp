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
#include <math.h>
#include <chrono>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop 

#include "CommandLineParser.h"
#define PI 3.14159265
using namespace cv;
using namespace std;


void convertRGBtoGray(Mat &output, const Mat &input);
void GaussianFilter(Mat &output, Mat &input);

//int preprocess();
//int verify_results(Mat &outputimage);

Mat preprocess(const Mat& inputImage, int scaleFactor);
Mat compute_reference_image(const Mat& inputImage);
bool verify_results(Mat &outputimage, Mat& refimage, int tol);
void WriteImageAsText(const Mat& outputimage, const std::string& txtFile);
void InitializeCmdLineParser(VelocityBench::CommandLineParser &parser);
