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


#include "ewCudaKernels.hpp"
#include "ewGpuNode.hpp"
#include <sycl.hpp>
#include <cmath>

SYCL_EXTERNAL __attribute__((always_inline)) void waveUpdate(KernelData data, sycl::nd_item<2> item_ct1)
{
    Params &dp = data.params;

    int   i  = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + dp.iMin;
    int   j  = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1) + dp.jMin;
    int   ij = data.idx(i, j);
    float absH;

    /* maybe unnecessary if controlled from outside */
    if (i <= dp.iMax && j <= dp.jMax && data.d[ij] != 0) {

        float hh = data.h[ij] - data.cR1[ij] * (data.fM[ij] - data.fM[data.le(ij)] + data.fN[ij] * data.cR6[j] - data.fN[data.dn(ij)] * data.cR6[j - 1]);

        absH = sycl::fabs(hh);

        if (absH < dp.sshZeroThreshold) {
            hh = 0.f;
        } else if (hh > data.hMax[ij]) {
            data.hMax[ij] = hh;
            // hMax[ij] = fmaxf(hMax[ij],h[ij]);
        }

        if (dp.sshArrivalThreshold && data.tArr[ij] < 0 && absH > dp.sshArrivalThreshold)
            data.tArr[ij] = dp.mTime;
        data.h[ij] = hh;
    }
}

SYCL_EXTERNAL __attribute__((always_inline)) void fluxUpdate(KernelData data, sycl::nd_item<2> item_ct1)
{
    Params &dp = data.params;

    int i  = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + dp.iMin;
    int j  = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1) + dp.jMin;
    int ij = data.idx(i, j);

    if (i <= dp.iMax && j <= dp.jMax && data.d[ij] != 0) {

        float hh = data.h[ij];

        if (data.d[data.ri(ij)] != 0) {
            data.fM[ij] = data.fM[ij] - data.cR2[ij] * (data.h[data.ri(ij)] - hh);
        }

        if (data.d[data.up(ij)] != 0)
            data.fN[ij] = data.fN[ij] - data.cR4[ij] * (data.h[data.up(ij)] - hh);
    }
}

#define SQR(x) powf(x, 2)
#ifdef USE_STD_MATH
#define SQRT(x) std::sqrt(x)
#else
#define SQRT(x) sycl::sqrt(x)
#endif

SYCL_EXTERNAL __attribute__((always_inline)) void waveBoundary(KernelData data, sycl::nd_item<1> item_ct1)
{
    KernelData &dt = data;
    Params     &dp = data.params;

    int id = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + 2;
    int ij;

    if (id <= dp.nI - 1) {
        ij       = dt.idx(id, 1);
        dt.h[ij] = SQRT(SQR(dt.fN[ij]) + 0.25f * SQR((dt.fM[ij] + dt.fM[dt.le(ij)]))) * dt.cB1[id - 1];
        if (dt.fN[ij] > 0)
            dt.h[ij] = -dt.h[ij];
    }

    if (id <= dp.nJ - 1) {
        ij       = dt.idx(1, id);
        dt.h[ij] = SQRT(SQR(dt.fM[ij]) + 0.25f * SQR((dt.fN[ij] + dt.fN[dt.dn(ij)]))) * dt.cB2[id - 1];
        if (dt.fM[ij] > 0)
            dt.h[ij] = -dt.h[ij];
    }

    if (id <= dp.nI - 1) {
        ij       = dt.idx(id, dp.nJ);
        dt.h[ij] = SQRT(SQR(dt.fN[dt.dn(ij)]) + 0.25f * SQR((dt.fM[ij] + dt.fM[dt.dn(ij)]))) * dt.cB3[id - 1];
        if (dt.fN[dt.dn(ij)] < 0)
            dt.h[ij] = -dt.h[ij];
    }

    if (id <= dp.nJ - 1) {
        ij       = dt.idx(dp.nI, id);
        dt.h[ij] = SQRT(SQR(dt.fM[dt.le(ij)]) + 0.25f * SQR((dt.fN[ij] + dt.fN[dt.dn(ij)]))) * dt.cB4[id - 1];
        if (dt.fM[dt.le(ij)] < 0)
            dt.h[ij] = -dt.h[ij];
    }

    if (id == 2) {
        ij       = dt.idx(1, 1);
        dt.h[ij] = SQRT(SQR(dt.fM[ij]) + SQR(dt.fN[ij])) * dt.cB1[0];
        if (dt.fN[ij] > 0)
            dt.h[ij] = -dt.h[ij];

        ij       = dt.idx(dp.nI, 1);
        dt.h[ij] = SQRT(SQR(dt.fM[dt.le(ij)]) + SQR(dt.fN[ij])) * dt.cB1[dp.nI - 1];
        if (dt.fN[ij] > 0)
            dt.h[ij] = -dt.h[ij];

        ij       = dt.idx(1, dp.nJ);
        dt.h[ij] = SQRT(SQR(dt.fM[ij]) + SQR(dt.fN[dt.dn(ij)])) * dt.cB3[0];
        if (dt.fN[dt.dn(ij)] < 0)
            dt.h[ij] = -dt.h[ij];

        ij       = dt.idx(dp.nI, dp.nJ);
        dt.h[ij] = SQRT(SQR(dt.fM[dt.le(ij)]) + SQR(dt.fN[dt.dn(ij)])) * dt.cB3[dp.nI - 1];
        if (dt.fN[dt.dn(ij)] < 0)
            dt.h[ij] = -dt.h[ij];
    }
}

SYCL_EXTERNAL __attribute__((always_inline)) void fluxBoundary(KernelData data, sycl::nd_item<1> item_ct1)
{
    KernelData &dt = data;
    Params     &dp = data.params;

    int id = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + 1;
    int ij;

    if (id <= dp.nI - 1) {
        ij        = dt.idx(id, 1);
        dt.fM[ij] = dt.fM[ij] - dt.cR2[ij] * (dt.h[dt.ri(ij)] - dt.h[ij]);
    }

    if (id <= dp.nJ) {
        ij        = dt.idx(1, id);
        dt.fM[ij] = dt.fM[ij] - dt.cR2[ij] * (dt.h[dt.ri(ij)] - dt.h[ij]);
    }

    if (id <= dp.nI - 1) {
        ij        = dt.idx(id, dp.nJ);
        dt.fM[ij] = dt.fM[ij] - dt.cR2[ij] * (dt.h[dt.ri(ij)] - dt.h[ij]);
    }

    if (id <= dp.nJ - 1) {
        ij        = dt.idx(1, id);
        dt.fN[ij] = dt.fN[ij] - dt.cR4[ij] * (dt.h[dt.up(ij)] - dt.h[ij]);
    }

    if (id <= dp.nI) {
        ij        = dt.idx(id, 1);
        dt.fN[ij] = dt.fN[ij] - dt.cR4[ij] * (dt.h[dt.up(ij)] - dt.h[ij]);
    }

    if (id <= dp.nJ - 1) {
        ij        = dt.idx(dp.nI, id);
        dt.fN[ij] = dt.fN[ij] - dt.cR4[ij] * (dt.h[dt.up(ij)] - dt.h[ij]);
    }
}

SYCL_EXTERNAL __attribute__((always_inline)) void gridExtend(KernelData data, sycl::nd_item<1> item_ct1)
{
    Params &dp = data.params;

    int id = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0) + 1;

    if (id >= dp.jMin && id <= dp.jMax) {

        if (sycl::fabs(data.h[data.idx(dp.iMin + 2, id)]) > dp.sshClipThreshold) {
            sycl::atomic_ref<int, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space>(data.g_MinMax->x())++;
        }

        if (sycl::fabs(data.h[data.idx(dp.iMax - 2, id)]) > dp.sshClipThreshold) {
            sycl::atomic_ref<int, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space>(data.g_MinMax->y())++;
        }
    }

    if (id >= dp.iMin && id <= dp.iMax) {

        if (sycl::fabs(data.h[data.idx(id, dp.jMin + 2)]) > dp.sshClipThreshold) {
            sycl::atomic_ref<int, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space>(data.g_MinMax->z())++;
        }

        if (sycl::fabs(data.h[data.idx(id, dp.jMax - 2)]) > dp.sshClipThreshold) {
            sycl::atomic_ref<int, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space>(data.g_MinMax->w())++;
        }
    }
}
