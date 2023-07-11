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

#ifndef CUDASIFTD_H
#define CUDASIFTD_H

#define NUM_SCALES 5

// Scale down thread block width
#define SCALEDOWN_W 64 // 60

// Scale down thread block height
#define SCALEDOWN_H 16 // 8

// Scale up thread block width
#define SCALEUP_W 64

// Scale up thread block height
#define SCALEUP_H 8

// Find point thread block width
#define MINMAX_W 30 // 32

// Find point thread block height
#define MINMAX_H 8 // 16

// Laplace thread block width
#define LAPLACE_W 128 // 56

// Laplace rows per thread
#define LAPLACE_H 4

// Number of laplace scales
#define LAPLACE_S (NUM_SCALES + 3)

// Laplace filter kernel radius
#define LAPLACE_R 4

#define LOWPASS_W 24 // 56
#define LOWPASS_H 32 // 16
#define LOWPASS_R 4

//====================== Number of threads ====================//
// ScaleDown:               SCALEDOWN_W + 4
// LaplaceMulti:            (LAPLACE_W+2*LAPLACE_R)*LAPLACE_S
// FindPointsMulti:         MINMAX_W + 2
// ComputeOrientations:     128
// ExtractSiftDescriptors:  256

//====================== Number of blocks ====================//
// ScaleDown:               (width/SCALEDOWN_W) * (height/SCALEDOWN_H)
// LaplceMulti:             (width+2*LAPLACE_R)/LAPLACE_W * height
// FindPointsMulti:         (width/MINMAX_W)*NUM_SCALES * (height/MINMAX_H)
// ComputeOrientations:     numpts
// ExtractSiftDescriptors:  numpts

#endif
