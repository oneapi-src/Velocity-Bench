//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//

// Modifications Copyright (C) 2023 Intel Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
// OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
// OR OTHER DEALINGS IN THE SOFTWARE.

// SPDX-License-Identifier: MIT

#include "hip/hip_runtime.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <hip/hip_runtime.h>

#include "cudautils.h"
#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftD.h"
#include "cudaSiftH.h"

#include "cudaSiftD.cpp"

void InitCuda(int devNum)
{
  int nDevices;
  hipGetDeviceCount(&nDevices);
  if (!nDevices)
  {
    std::cerr << "No CUDA devices available" << std::endl;
    return;
  }
  devNum = std::min(nDevices - 1, devNum);
  deviceInit(devNum);
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, devNum);
  printf("Device Number: %d\n", devNum);
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1000);
  printf("  Clock Freq (MHz): %d\n", prop.clockRate / 1000);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %.1f\n\n",
         2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

float *AllocSiftTempMemory(int width, int height, int numOctaves, float &time, bool scaleUp)
{
  const int nd = NUM_SCALES + 3;
  int w = width * (scaleUp ? 2 : 1);
  int h = height * (scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int size = h * p;         // image sizes
  int sizeTmp = nd * h * p; // laplace buffer sizes
  for (int i = 0; i < numOctaves; i++)
  {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h * p;
    sizeTmp += nd * h * p;
  }
  float *memoryTmp = NULL;
  size_t pitch;
  size += sizeTmp;

#ifdef DEVICE_TIMER
  auto start_malloc = std::chrono::steady_clock::now();
#endif
  hipMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float));
  hipDeviceSynchronize();

#ifdef DEVICE_TIMER
  auto stop_malloc = std::chrono::steady_clock::now();
  // printf("Malloc time for memoryTmp =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_malloc - start_malloc).count());
  time += std::chrono::duration<float, std::micro>(stop_malloc - start_malloc).count();
#endif
  return memoryTmp;
}

void FreeSiftTempMemory(float *memoryTmp)
{
  if (memoryTmp)
    hipFree(memoryTmp);
}

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur,
                 float thresh, float &totTime, float lowestScale, bool scaleUp, float *tempMemory)
{
  unsigned int *d_PointCounterAddr;

#ifdef DEVICE_TIMER
  auto start_memcpy = std::chrono::steady_clock::now();
#endif
  hipGetSymbolAddress((void **)&d_PointCounterAddr, HIP_SYMBOL(d_PointCounter));
  hipMemset(d_PointCounterAddr, 0, (8 * 2 + 1) * sizeof(int));
  hipMemcpyToSymbol(HIP_SYMBOL(d_MaxNumPoints), &siftData.maxPts, sizeof(int));
  hipDeviceSynchronize();

#ifdef DEVICE_TIMER
  auto stop_memcpy = std::chrono::steady_clock::now();
  totTime += std::chrono::duration<float, std::micro>(stop_memcpy - start_memcpy).count();
#endif

  const int nd = NUM_SCALES + 3;
  int w = img.width * (scaleUp ? 2 : 1);
  int h = img.height * (scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int width = w, height = h;
  int size = h * p;         // image sizes
  int sizeTmp = nd * h * p; // laplace buffer sizes
  for (int i = 0; i < numOctaves; i++)
  {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h * p;
    sizeTmp += nd * h * p;
  }
  float *memoryTmp = tempMemory;
  size += sizeTmp;
  if (!tempMemory)
  {
    size_t pitch;

#ifdef DEVICE_TIMER
    auto start_malloc = std::chrono::steady_clock::now();
#endif
    safeCall(hipMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float)));
    hipDeviceSynchronize();

#ifdef DEVICE_TIMER
    auto stop_malloc = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_malloc - start_malloc).count();
#endif
  }
  float *memorySub = memoryTmp + sizeTmp;

  CudaImage lowImg;
  lowImg.Allocate(width, height, iAlignUp(width, 128), false, totTime, memorySub);
  if (!scaleUp)
  {
    float kernel[8 * 12 * 16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);

#ifdef DEVICE_TIMER
    auto start_memcpy1 = std::chrono::steady_clock::now();
#endif
    hipMemcpyToSymbol(HIP_SYMBOL(d_LaplaceKernel), kernel, 8 * 12 * 16 * sizeof(float), 0, hipMemcpyHostToDevice);
    hipDeviceSynchronize();

#ifdef DEVICE_TIMER
    auto stop_memcpy1 = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy1 - start_memcpy1).count();
#endif
    LowPass(lowImg, img, fmax(initBlur, 0.001f), totTime);
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp,
                    memorySub + height * iAlignUp(width, 128), totTime);

#ifdef DEVICE_TIMER
    auto start_memcpy2 = std::chrono::steady_clock::now();
#endif
    safeCall(hipMemcpy(&siftData.numPts, &d_PointCounterAddr[2 * numOctaves], sizeof(int), hipMemcpyDeviceToHost));
    hipDeviceSynchronize();

#ifdef DEVICE_TIMER
    auto stop_memcpy2 = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy2 - start_memcpy2).count();
#endif
    siftData.numPts = (siftData.numPts < siftData.maxPts ? siftData.numPts : siftData.maxPts);
  }
  else
  {
    CudaImage upImg;
    upImg.Allocate(width, height, iAlignUp(width, 128), false, totTime, memoryTmp);
    ScaleUp(upImg, img, totTime);
    LowPass(lowImg, upImg, max(initBlur, 0.001f), totTime);
    float kernel[8 * 12 * 16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);

#ifdef DEVICE_TIMER
    auto start_memcpy3 = std::chrono::steady_clock::now();
#endif
    hipMemcpyToSymbol(HIP_SYMBOL(d_LaplaceKernel), kernel, 8 * 12 * 16 * sizeof(float), 0, hipMemcpyHostToDevice);
    hipDeviceSynchronize();

#ifdef DEVICE_TIMER
    auto stop_memcpy3 = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy3 - start_memcpy3).count();
#endif
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale * 2.0f, 1.0f, memoryTmp,
                    memorySub + height * iAlignUp(width, 128), totTime);

#ifdef DEVICE_TIMER
    auto start_memcpy4 = std::chrono::steady_clock::now();
#endif
    safeCall(hipMemcpy(&siftData.numPts, &d_PointCounterAddr[2 * numOctaves], sizeof(int), hipMemcpyDeviceToHost));
    hipDeviceSynchronize();

#ifdef DEVICE_TIMER
    auto stop_memcpy4 = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy4 - start_memcpy4).count();
#endif
    siftData.numPts = (siftData.numPts < siftData.maxPts ? siftData.numPts : siftData.maxPts);
    RescalePositions(siftData, 0.5f, totTime);
  }

  if (!tempMemory)
    safeCall(hipFree(memoryTmp));
#ifdef MANAGEDMEM
  safeCall(hipDeviceSynchronize());
#else
  if (siftData.h_data)
  {
#ifdef DEVICE_TIMER
    auto start_memcpy5 = std::chrono::steady_clock::now();
#endif
    safeCall(hipMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint) * siftData.numPts, hipMemcpyDeviceToHost));
    hipDeviceSynchronize();

#ifdef DEVICE_TIMER
    auto stop_memcpy5 = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy5 - start_memcpy5).count();
    printf("Total time for sift extraction = %0.2f us %d \n\n", totTime, siftData.numPts);
#endif
  }
#endif
  printf("Number of Points after sift extraction =  %d\n\n", siftData.numPts);
}

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale,
                    float subsampling, float *memoryTmp, float *memorySub, float &totTime)
{
  int w = img.width;
  int h = img.height;
  if (numOctaves > 1)
  {
    CudaImage subImg;
    int p = iAlignUp(w / 2, 128);
    subImg.Allocate(w / 2, h / 2, p, false, totTime, memorySub);
    ScaleDown(subImg, img, 0.5f, totTime);
    float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
    ExtractSiftLoop(siftData, subImg, numOctaves - 1, totInitBlur, thresh, lowestScale, subsampling * 2.0f,
                    memoryTmp, memorySub + (h / 2) * p, totTime);
  }

  ExtractSiftOctave(siftData, img, numOctaves, thresh, lowestScale, subsampling, memoryTmp, totTime);
  return 0;
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh,
                       float lowestScale, float subsampling, float *memoryTmp, float &totTime)
{
  const int nd = NUM_SCALES + 3;
  CudaImage diffImg[nd];
  int w = img.width;
  int h = img.height;
  int p = iAlignUp(w, 128);
  for (int i = 0; i < nd - 1; i++)
    diffImg[i].Allocate(w, h, p, false, totTime, memoryTmp + i * p * h);

  float baseBlur = pow(2.0f, -1.0f / NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
  LaplaceMulti(img, diffImg, octave, totTime);
  FindPointsMulti(diffImg, siftData, thresh, 10.0f, 1.0f / NUM_SCALES, lowestScale / subsampling, subsampling, octave, totTime);
  ComputeOrientations(img, siftData, octave, totTime);
  ExtractSiftDescriptors(img.d_data, img.pitch, siftData, subsampling, octave, totTime);
}

void InitSiftData(SiftData &data, float &time, int num, bool host, bool dev)
{
  data.numPts = 0;
  data.maxPts = num;
  int sz = sizeof(SiftPoint) * num;
  data.h_data = NULL;
  if (host)
    data.h_data = (SiftPoint *)malloc(sz);
  data.d_data = NULL;
  if (dev)
  {
#ifdef DEVICE_TIMER
    auto start_malloc = std::chrono::steady_clock::now();
#endif
    safeCall(hipMalloc((void **)&data.d_data, sz));
    hipDeviceSynchronize();

#ifdef DEVICE_TIMER
    auto stop_malloc = std::chrono::steady_clock::now();
    time += std::chrono::duration<float, std::micro>(stop_malloc - start_malloc).count();
#endif
  }
}

void FreeSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
  safeCall(hipFree(data.m_data));
#else
  if (data.d_data != NULL)
    safeCall(hipFree(data.d_data));
  data.d_data = NULL;
  if (data.h_data != NULL)
    free(data.h_data);
#endif
  data.numPts = 0;
  data.maxPts = 0;
}

void PrintSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
  SiftPoint *h_data = data.m_data;
#else
  SiftPoint *h_data = data.h_data;
  if (data.h_data == NULL)
  {
    h_data = (SiftPoint *)malloc(sizeof(SiftPoint) * data.maxPts);
    safeCall(hipMemcpy(h_data, data.d_data, sizeof(SiftPoint) * data.numPts, hipMemcpyDeviceToHost));
    hipDeviceSynchronize();
    data.h_data = h_data;
  }
#endif
  for (int i = 0; i < data.numPts; i++)
  {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    printf("score        = %.2f\n", h_data[i].score);
    float *siftData = (float *)&h_data[i].data;
    for (int j = 0; j < 8; j++)
    {
      if (j == 0)
        printf("data = ");
      else
        printf("       ");
      for (int k = 0; k < 16; k++)
        if (siftData[j + 8 * k] < 0.05)
          printf(" .   ");
        else
          printf("%.2f ", siftData[j + 8 * k]);
      printf("\n");
    }
  }
  printf("Number of available points: %d\n", data.numPts);
  printf("Number of allocated points: %d\n", data.maxPts);
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double ScaleDown(CudaImage &res, CudaImage &src, float variance, float &totTime)
{
  static float oldVariance = -1.0f;
  if (res.d_data == NULL || src.d_data == NULL)
  {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
  if (oldVariance != variance)
  {
    float h_Kernel[5];
    float kernelSum = 0.0f;
    for (int j = 0; j < 5; j++)
    {
      h_Kernel[j] = (float)expf(-(double)(j - 2) * (j - 2) / 2.0 / variance);
      kernelSum += h_Kernel[j];
    }
    for (int j = 0; j < 5; j++)
      h_Kernel[j] /= kernelSum;

#ifdef DEVICE_TIMER
    auto start_memcpy = std::chrono::steady_clock::now();
#endif
    safeCall(hipMemcpyToSymbol(HIP_SYMBOL(d_ScaleDownKernel), h_Kernel, 5 * sizeof(float)));
    hipDeviceSynchronize();

#ifdef DEVICE_TIMER
    auto stop_memcpy = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy - start_memcpy).count();
#endif
    oldVariance = variance;
  }
#if 0
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4, SCALEDOWN_H + 4);
  hipLaunchKernelGGL(ScaleDownDenseShift, blocks, threads, 0, 0, res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#else
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  hipLaunchKernelGGL(ScaleDown, blocks, threads, 0, 0, res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
  hipDeviceSynchronize();

#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
#endif
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

double ScaleUp(CudaImage &res, CudaImage &src, float &totTime)
{
  if (res.d_data == NULL || src.d_data == NULL)
  {
    printf("ScaleUp: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
  dim3 threads(SCALEUP_W / 2, SCALEUP_H / 2);

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  hipLaunchKernelGGL(ScaleUp, blocks, threads, 0, 0, res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
  hipDeviceSynchronize();

#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("ScaleUp time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif

  checkMsg("ScaleUp() execution failed\n");
  return 0.0;
}

double ComputeOrientations(CudaImage &src, SiftData &siftData, int octave, float &totTime)
{
  dim3 blocks(512);
  dim3 threads(256);

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  hipLaunchKernelGGL(ComputeOrientationsCONSTNew, blocks, threads, 0, 0, src.d_data, src.width, src.pitch, src.height, siftData.d_data, octave);
  hipDeviceSynchronize();

#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("ComputeOrientationsCONSTNew time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
  checkMsg("ComputeOrientations() execution failed\n");
  return 0.0;
}

double ExtractSiftDescriptors(float *texObj, int pitch, SiftData &siftData, float subsampling, int octave, float &totTime)
{
  dim3 blocks(512);
  dim3 threads(16, 8);

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  hipLaunchKernelGGL(ExtractSiftDescriptorsCONSTNew, blocks, threads, 0, 0, texObj, pitch, siftData.d_data, subsampling, octave);
  hipDeviceSynchronize();

#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("ExtractSiftDescriptorsCONSTNew time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  return 0.0;
}

double RescalePositions(SiftData &siftData, float scale, float &totTime)
{
  dim3 blocks(iDivUp(siftData.numPts, 64));
  dim3 threads(64);

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  hipLaunchKernelGGL(RescalePositions, blocks, threads, 0, 0, siftData.d_data, siftData.numPts, scale);
  hipDeviceSynchronize();

#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("RescalePositions time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif

  checkMsg("RescapePositions() execution failed\n");
  return 0.0;
}

double LowPass(CudaImage &res, CudaImage &src, float scale, float &totTime)
{
  float kernel[2 * LOWPASS_R + 1];
  static float oldScale = -1.0f;
  if (scale != oldScale)
  {
    float kernelSum = 0.0f;
    float ivar2 = 1.0f / (2.0f * scale * scale);
    for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
    {
      kernel[j + LOWPASS_R] = (float)expf(-(double)j * j * ivar2);
      kernelSum += kernel[j + LOWPASS_R];
    }
    for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
      kernel[j + LOWPASS_R] /= kernelSum;

#ifdef DEVICE_TIMER
    auto start_memcpy_1 = std::chrono::steady_clock::now();
#endif
    safeCall(hipMemcpyToSymbol(HIP_SYMBOL(d_LowPassKernel), kernel, (2 * LOWPASS_R + 1) * sizeof(float)));
    hipDeviceSynchronize();

#ifdef DEVICE_TIMER
    auto stop_memcpy_1 = std::chrono::steady_clock::now();
    totTime += std::chrono::duration<float, std::micro>(stop_memcpy_1 - start_memcpy_1).count();
#endif

    oldScale = scale;
  }
  int width = res.width;
  int pitch = res.pitch;
  int height = res.height;
  dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H)); //[80,34,1]
#if 1
  dim3 threads(LOWPASS_W + 2 * LOWPASS_R, 4); //[32,4,1]

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  hipLaunchKernelGGL(LowPassBlock, blocks, threads, 0, 0, src.d_data, res.d_data, width, pitch, height);
  hipDeviceSynchronize();

#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("LowPassBlock time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
#else
  dim3 threads(LOWPASS_W + 2 * LOWPASS_R, LOWPASS_H);
  hipLaunchKernelGGL(LowPass, blocks, threads, 0, 0, src.d_data, res.d_data, width, pitch, height);
#endif
  checkMsg("LowPass() execution failed\n");
  return 0.0;
}

//==================== Multi-scale functions ===================//

void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel)
{
  if (numOctaves > 1)
  {
    float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
    PrepareLaplaceKernels(numOctaves - 1, totInitBlur, kernel);
  }
  float scale = pow(2.0f, -1.0f / NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
  for (int i = 0; i < NUM_SCALES + 3; i++)
  {
    float kernelSum = 0.0f;
    float var = scale * scale - initBlur * initBlur;
    for (int j = 0; j <= LAPLACE_R; j++)
    {
      kernel[numOctaves * 12 * 16 + 16 * i + j] = (float)expf(-(double)j * j / 2.0 / var);
      kernelSum += (j == 0 ? 1 : 2) * kernel[numOctaves * 12 * 16 + 16 * i + j];
    }
    for (int j = 0; j <= LAPLACE_R; j++)
      kernel[numOctaves * 12 * 16 + 16 * i + j] /= kernelSum;
    scale *= diffScale;
  }
}

double LaplaceMulti(CudaImage &baseImage, CudaImage *results, int octave, float &totTime)
{
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;
#if 1
  dim3 threads(LAPLACE_W + 2 * LAPLACE_R);       //(136)
  dim3 blocks(iDivUp(width, LAPLACE_W), height); //(15)

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  hipLaunchKernelGGL(LaplaceMultiMem, blocks, threads, 0, 0, baseImage.d_data, results[0].d_data, width, pitch, height, octave);
  hipDeviceSynchronize();

#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("LaplaceMultiMem time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count());
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
#endif
  checkMsg("LaplaceMulti() execution failed\n");
  return 0.0;
}

double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor,
                       float lowestScale, float subsampling, int octave, float &totTime)
{
  if (sources->d_data == NULL)
  {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;
#if 1
  dim3 blocks(iDivUp(w, MINMAX_W) * NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2);

#ifdef DEVICE_TIMER
  auto start_kernel = std::chrono::steady_clock::now();
#endif
  hipLaunchKernelGGL(FindPointsMultiNew, blocks, threads, 0, 0, sources->d_data, siftData.d_data, w, p, h, subsampling,
                     lowestScale, thresh, factor, edgeLimit, octave);
  hipDeviceSynchronize();
#ifdef DEVICE_TIMER
  auto stop_kernel = std::chrono::steady_clock::now();
  // printf("FindPointsMultiNew time =          %.2f us\n", std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count())
  totTime += std::chrono::duration<float, std::micro>(stop_kernel - start_kernel).count();
#endif
#endif
  checkMsg("FindPointsMulti() execution failed\n");
  return 0.0;
}
