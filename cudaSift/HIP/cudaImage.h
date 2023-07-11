//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
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

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

#include <hip/hip_runtime.h>
#define OCML_BASIC_ROUNDED_OPERATIONS

class CudaImage
{
public:
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  float *t_data;
  bool d_internalAlloc;
  bool h_internalAlloc;

public:
  CudaImage();
  CudaImage(const CudaImage&) = delete;
  CudaImage& operator=(const CudaImage&) = delete;
  ~CudaImage();
  void Allocate(int width, int height, int pitch, bool withHost, float &time, float *devMem = NULL, float *hostMem = NULL);
  double Download(float &time);
  double Readback();
  double InitTexture();
  double CopyToTexture(CudaImage &dst, bool host);
};

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);
void StartTimer(unsigned int *hTimer);
double StopTimer(unsigned int hTimer);

#endif // CUDAIMAGE_H
