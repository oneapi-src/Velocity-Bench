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

#include "common.hpp"
#include <fstream>
#undef CPP_MODULE
#define CPP_MODULE "COMN"
#include "Logging.h"
//double dImageScale(11.0);
double dImageScale(5.0);

void convertRGBtoGray(Mat &output, const Mat &input)
{
    int rows = input.rows;
    int cols = input.cols;
    for(int i = 0; i < rows*cols; i++)
        output.data[i] = input.data[i*3];
    return;
}

void GaussianFilter(Mat &output, Mat &input)
{
    float filter[3][3] = {{0.04491, 0.12210, 0.04491}, {0.12210, 0.33191, 0.12210}, {0.04491, 0.12210, 0.04491}};
    int rows = output.rows;
    int cols = output.cols;
    float value;
    for(int i = 1; i < rows; i++)
    {
        int index = i*cols+1;
        for(int j = index; j < index+cols; j++)
        {
            value = 0.0f;
            for(int k1 = -1; k1 <= 1; k1++)
            {
                int index1 = j+(k1*cols);
                for(int k2 = -1; k2 <= 1; k2++)
                    value += filter[k1+1][k2+1]*input.data[index1+k2];
            }
            output.data[j] = value;
        }
    }
    return;
}

void computeGradient_scalar(const Mat &input, Mat &strength)
{
    float gradientx[3][3] = {{-1.f, 0.f, 1.f}, {-2.f, 0.f, 2.f}, {-1.f, 0.f, 1.f}};
    float gradienty[3][3] = {{-1.f, -2.f, -1.f}, {0.f, 0.f, 0.f}, {1.f, 2.f, 1.f}};
    int rows = input.rows;
    int cols = input.cols;
    int gcols = strength.cols;
    float gradient_x, gradient_y;
    //float degrees;
    for(int i = 1; i < rows-1; i++)
    {
        for(int j = 1; j < cols-1; j++)
        {
            int oPixelPos = (i-1) * (gcols) + (j-1);
            gradient_x = gradient_y = 0.0f;
            for(int k1 = -1; k1 <= 1; k1++)
            {
                for(int k2 = -1; k2 <=1; k2++)
                {
                    unsigned int iPixelPos = ( i + k1 ) * cols + ( j + k2 );
                    //unsigned int coefPos = ( k1 + 1 ) * 9 + ( k2 + 1 );
                    gradient_x += gradientx[k1+1][k2+1] * input.data[iPixelPos];
                    gradient_y += gradienty[k1+1][k2+1] * input.data[iPixelPos];
                }
            }
            strength.data[oPixelPos] = sqrtf(powf(gradient_x, 2.f) + powf(gradient_y, 2.f));
        }
    }
    imwrite("../res/scalar.bmp", strength);
    return;
}

/*
int preprocess()
{
    Mat inputimage;
    //string image_file("../res/test.bmp");
    string image_file("../res/silverfalls.bmp");
    inputimage = imread(image_file, IMREAD_COLOR);
    cout << "image before scaling: number of rows: "<<inputimage.rows <<", columns: "<<inputimage.cols<< endl;
    //resize(inputimage, inputimage,Size(inputimage.rows * 10, inputimage.cols * 10));
    cv::resize(inputimage, inputimage, cv::Size(inputimage.cols * dImageScale, inputimage.rows * dImageScale));
    cout << "image after scaling: number of rows: "<<inputimage.rows <<", columns: "<<inputimage.cols<< endl;

    int rows = inputimage.rows;
    int cols = inputimage.cols;
    //int newrows = rows*3;
    //int newcols = cols*4;

    //resize(inputimage,inputimage,Size(inputimage.rows * 3, inputimage.cols * 4));	



    int newrows = rows+2;
    int newcols = cols+2;
    Mat outputimage(rows, cols, CV_8UC1);
    Mat scalarimage(rows, cols, CV_8UC1);
    Mat outputimage1(newrows, newcols, CV_8UC1);
    Mat outputimage2(newrows, newcols, CV_8UC1);
    memset(outputimage1.data, 0, rows*cols);
    memset(outputimage2.data, 0, rows*cols);
    if(!inputimage.data)
    {
        cout<<"failed to open the image\n";
        return -1;
    }
    //Step 1 RGB to Gray Scale Conversion
    convertRGBtoGray(outputimage, inputimage);
    //Get padding around the image for applying Gaussian and other filters
    for(int i = 0; i < rows; i++)
    {
        int index1 = (i+1)*newcols+1;
        int index = i*cols;
        for(int j = 0; j < cols; j++)
            outputimage1.data[index1+j] = outputimage.data[index+j];
    }
    //Step 2 Gaussian Filter on Gray Scale Image
    GaussianFilter(outputimage2, outputimage1);


    imwrite("../res/input.bmp", outputimage2);

    computeGradient_scalar(outputimage2, scalarimage);
    return 0;
}
*/

Mat preprocess(const Mat& src, int scaleFactor)
{
    Mat inputimage(src);
    if(!inputimage.data)
        LOG_ERROR("Input image is empty.");
    if(scaleFactor < 1 || scaleFactor > 20)
        LOG_ERROR("Scale factor is out of range ( 1 - 20 ).");

    int rows = inputimage.rows;
    int cols = inputimage.cols;
    LOG("Input image: number of rows: " << rows);
    LOG("Input image: number of cols: " << cols);

    LOG("Upscaling input image in place by a factor of "<< scaleFactor);
    cv::resize(inputimage, inputimage, cv::Size(inputimage.cols*scaleFactor, inputimage.rows*scaleFactor));

    rows = inputimage.rows;
    cols = inputimage.cols;
    LOG("Scaled image: number of rows: " << rows);
    LOG("Scaled image: number of cols: " << cols);

    int newrows = rows+2;
    int newcols = cols+2;
    Mat outputimage(rows, cols, CV_8UC1);
    Mat scalarimage(rows, cols, CV_8UC1);
    Mat outputimage1(newrows, newcols, CV_8UC1);
    Mat outputimage2(newrows, newcols, CV_8UC1);
    memset(outputimage1.data, 0, rows*cols);
    memset(outputimage2.data, 0, rows*cols);
    if(!inputimage.data)
    {
        LOG_ERROR("failed to open the image\n");
        return outputimage;
    }
    //Step 1 RGB to Gray Scale Conversion
    convertRGBtoGray(outputimage, inputimage);
    //Get padding around the image for applying Gaussian and other filters
    for(int i = 0; i < rows; i++)
    {
        int index1 = (i+1)*newcols+1;
        int index = i*cols;
        for(int j = 0; j < cols; j++)
            outputimage1.data[index1+j] = outputimage.data[index+j];
    }
    //Step 2 Gaussian Filter on Gray Scale Image
    GaussianFilter(outputimage2, outputimage1);

    return outputimage2;
}

Mat compute_reference_image(const Mat& inputimage)
{
    Mat refimage = cv::Mat::zeros(inputimage.rows, inputimage.cols, CV_8UC1);
    computeGradient_scalar(inputimage, refimage);

    return refimage;
}

/*
int verify_results(Mat &outputimage){
    cout << "Running data verification."<<endl;

    //Mat inputimage;
    //inputimage = imread("../res/input.bmp", cv::IMREAD_UNCHANGED);
    Mat baseimage;
    //computeGradient(inputimage,baseimage);
    baseimage = imread("../res/scalar.bmp", cv::IMREAD_UNCHANGED);
    //Mat outputimage;
    //outputimage = imread("../res/cuda.jpg",cv::IMREAD_UNCHANGED);
    int rows = baseimage.rows;
    int cols = baseimage.cols;

    //std::cout<< "baseimage rows" <<baseimage.rows << "baseimage cols" << baseimage.cols << "\n";
    //std::cout<< "outputimage rows" <<outputimage.rows << "baseimage cols" << outputimage.cols << "\n";
    int diffpixel = 0;
    if (baseimage.empty())
    {
        cout << "failed to open the base image\n";
        return -1;
    }
    if (outputimage.empty())
    {
        cout << "failed to open the output image\n";
        return -1;
    }

    for(int i=0; i < rows; i++)
        for(int j=0; j < cols; j++)
        {
            int diff = baseimage.at<char>(i,j) - outputimage.at<char>(i,j);
            if (abs(diff) >=5){

                // cout << "different pixels at  :" <<i<<"," <<j << "   " << diff <<"\n";
                diffpixel++;

            }

        }

    double tolerance = ((double)diffpixel/(double) (rows*cols))*100;
    std::cout<< "diffpixels: " << diffpixel <<"\t" << "tolerance: " << tolerance <<"\n";
    if (tolerance > 10){
        cout << "The images differ with tolerance greater than 10%" <<"\n";
        cout << "Verification FAILED"<<endl;
        return -1;
    }
    else{
        cout << "The images are identical with tolerance less than 10%" << "\n";
        cout << "Verification is SUCCESSFUL"<<endl;
    }

    return 0;

}
*/

bool verify_results(Mat &outputimage, Mat& refimage, int tol)
{
    LOG("Running data verification.");

    int diffpixel = 0;
    if (refimage.empty())
    {
        LOG_ERROR("Data verification failed. Reference image is empty");
        return false;
    }
    LOG("   Valid Reference image found.");

    if (outputimage.empty())
    {
        LOG_ERROR("Data verification failed. Output image is empty");
        return false;
    }
    LOG("   Valid Output image found.");

    LOG("   Ref image dimensions: ("<<refimage.rows << ", "<<refimage.cols<<").")
    LOG("   Output image dimensions: ("<<outputimage.rows<< ", "<<outputimage.cols<<").")

    int rows = outputimage.rows;
    int cols = outputimage.cols;

    for(int i=0; i < rows; i++)
    {
        for(int j=0; j < cols; j++)
        {
            int diff = refimage.at<char>(i,j) - outputimage.at<char>(i,j);
            if (abs(diff) >=tol)
            {
                diffpixel++;
            }

        }
    }
    LOG("   Image diff completed.");

    double tolerance = ((double)diffpixel/(double) (rows*cols))*100;
    LOG("diffpixels: " << diffpixel <<"\t" << "tolerance: " << tolerance);
    if (tolerance > 10){
        LOG("The images differ with tolerance greater than 10%.");
        LOG("Verification FAILED");
        return false;
    }
    else{
        LOG("Verification is SUCCESSFUL");
    }

    return true;
}

void WriteImageAsText(const Mat& outputimage, const std::string& txtFile)
{
	int rows = outputimage.rows;
	int cols = outputimage.cols;
    ofstream myfile;
    myfile.open(txtFile);
    if(myfile.is_open()){
        for(int i=0; i< rows; i++)
            for(int j=0; j < cols; j++)
            {
                myfile << (int) outputimage.at<uchar>(i,j) << "\n";
            }
    }

    myfile.close();
}

void InitializeCmdLineParser(VelocityBench::CommandLineParser &parser)
{
    using namespace VelocityBench;
    bool bRequired(true);
    parser.AddSetting("-i", "input image", bRequired, "",VelocityBench::CommandLineParser::InputType_t::STRING,     1);
    parser.AddSetting("-o", "output image",         false, "./output.bmp",  VelocityBench::CommandLineParser::InputType_t::STRING,     1);
    parser.AddSetting("-f", "scale factor",         false, "5",        VelocityBench::CommandLineParser::InputType_t::INTEGER,    1);
    parser.AddSetting("-n", "# of iterations",      false, "1",        VelocityBench::CommandLineParser::InputType_t::INTEGER,    1);
    parser.AddSetting("-d", "device type (dpc++)",  false, "gpu",      VelocityBench::CommandLineParser::InputType_t::STRING,     1);
    parser.AddSetting("-tol", "tolerance",  false, "5",      VelocityBench::CommandLineParser::InputType_t::INTEGER,     1);
    parser.AddSetting("-v", "run verification",  false, "1", VelocityBench::CommandLineParser::InputType_t::INTEGER,     0);
    parser.AddSetting("-saveref", "save reference image",  false, "0", VelocityBench::CommandLineParser::InputType_t::INTEGER, 0);
}
