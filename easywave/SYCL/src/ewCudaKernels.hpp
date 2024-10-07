/*
 * Modifications Copyright (C) 2023 Intel Corporation
 * 
 * This Program is subject to the terms of the European Union Public License 1.2
 * 
 * If a copy of the license was not distributed with this file, you can obtain one at 
 * https://joinup.ec.europa.eu/sites/default/files/custom-page/attachment/2020-03/EUPL-1.2%20EN.txt
 * 
 * SPDX-License-Identifier: EUPL-1.2
 */


#ifndef EW_CUDA_KERNELS
#define EW_CUDA_KERNELS

#include <sycl/sycl.hpp>
#include "ewGpuNode.hpp"

#ifdef USE_INLINE_KERNELS
/* see below for rationale */
#undef SYCL_EXTERNAL
#define SYCL_EXTERNAL static
#endif /* USE_INLINE_KERNELS */

SYCL_EXTERNAL __attribute__((always_inline)) void waveUpdate(KernelData data, sycl::nd_item<2> item_ct1);
SYCL_EXTERNAL __attribute__((always_inline)) void waveBoundary(KernelData data, sycl::nd_item<1> item_ct1);
SYCL_EXTERNAL __attribute__((always_inline)) void fluxUpdate(KernelData data, sycl::nd_item<2> item_ct1);
SYCL_EXTERNAL __attribute__((always_inline)) void fluxBoundary(KernelData data, sycl::nd_item<1> item_ct1);
SYCL_EXTERNAL __attribute__((always_inline)) void gridExtend(KernelData data, sycl::nd_item<1> item_ct1);

#ifdef USE_AMD_BACKEND
#include "ewCudaKernels.cpp"
#endif // llvm-amd backend has a bug

#endif
