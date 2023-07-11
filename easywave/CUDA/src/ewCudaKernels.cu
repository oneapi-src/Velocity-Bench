/*
 * EasyWave - A realtime tsunami simulation program with GPU support.
 * Copyright (C) 2014  Andrey Babeyko, Johannes Spazier
 * GFZ German Research Centre for Geosciences (http://www.gfz-potsdam.de)
 *
 * Parts of this program (especially the GPU extension) were developed
 * within the context of the following publicly funded project:
 * - TRIDEC, EU 7th Framework Programme, Grant Agreement 258723
 *   (http://www.tridec-online.eu)
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent versions of the EUPL (the "Licence"),
 * complemented with the following provision: For the scientific transparency
 * and verification of results obtained and communicated to the public after
 * using a modified version of the work, You (as the recipient of the source
 * code and author of this modified version, used to produce the published
 * results in scientific communications) commit to make this modified source
 * code available in a repository that is easily and freely accessible for a
 * duration of five years after the communication of the obtained results.
 *
 * You may not use this work except in compliance with the Licence.
 *
 * You may obtain a copy of the Licence at:
 * https://joinup.ec.europa.eu/software/page/eupl
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 */

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

#include "ewGpuNode.cuh"
#include "ewCudaKernels.cuh"

__global__ void runWaveUpdateKernel(KernelData data)
{

    Params &dp = data.params;

    int   i  = blockIdx.y * blockDim.y + threadIdx.y + dp.iMin;
    int   j  = blockIdx.x * blockDim.x + threadIdx.x + dp.jMin;
    int   ij = data.idx(i, j);
    float absH;

    /* maybe unnecessary if controlled from outside */
    if (i <= dp.iMax && j <= dp.jMax && data.d[ij] != 0) {

        float hh = data.h[ij] - data.cR1[ij] * (data.fM[ij] - data.fM[data.le(ij)] + data.fN[ij] * data.cR6[j] - data.fN[data.dn(ij)] * data.cR6[j - 1]);

        absH = fabsf(hh);

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

__global__ void runFluxUpdateKernel(KernelData data)
{

    Params &dp = data.params;

    int i  = blockIdx.y * blockDim.y + threadIdx.y + dp.iMin;
    int j  = blockIdx.x * blockDim.x + threadIdx.x + dp.jMin;
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

__global__ void runWaveBoundaryKernel(KernelData data)
{

    KernelData &dt = data;
    Params     &dp = data.params;

    int id = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int ij;

    if (id <= dp.nI - 1) {
        ij       = dt.idx(id, 1);
        dt.h[ij] = sqrtf(SQR(dt.fN[ij]) + 0.25f * SQR((dt.fM[ij] + dt.fM[dt.le(ij)]))) * dt.cB1[id - 1];
        if (dt.fN[ij] > 0)
            dt.h[ij] = -dt.h[ij];
    }

    if (id <= dp.nI - 1) {
        ij       = dt.idx(id, dp.nJ);
        dt.h[ij] = sqrtf(SQR(dt.fN[dt.dn(ij)]) + 0.25f * SQR((dt.fM[ij] + dt.fM[dt.dn(ij)]))) * dt.cB3[id - 1];
        if (dt.fN[dt.dn(ij)] < 0)
            dt.h[ij] = -dt.h[ij];
    }

    if (id <= dp.nJ - 1) {
        ij       = dt.idx(1, id);
        dt.h[ij] = sqrtf(SQR(dt.fM[ij]) + 0.25f * SQR((dt.fN[ij] + dt.fN[dt.dn(ij)]))) * dt.cB2[id - 1];
        if (dt.fM[ij] > 0)
            dt.h[ij] = -dt.h[ij];
    }

    if (id <= dp.nJ - 1) {
        ij       = dt.idx(dp.nI, id);
        dt.h[ij] = sqrtf(SQR(dt.fM[dt.le(ij)]) + 0.25f * SQR((dt.fN[ij] + dt.fN[dt.dn(ij)]))) * dt.cB4[id - 1];
        if (dt.fM[dt.le(ij)] < 0)
            dt.h[ij] = -dt.h[ij];
    }

    if (id == 2) {
        ij       = dt.idx(1, 1);
        dt.h[ij] = sqrtf(SQR(dt.fM[ij]) + SQR(dt.fN[ij])) * dt.cB1[0];
        if (dt.fN[ij] > 0)
            dt.h[ij] = -dt.h[ij];

        ij       = dt.idx(dp.nI, 1);
        dt.h[ij] = sqrtf(SQR(dt.fM[dt.le(ij)]) + SQR(dt.fN[ij])) * dt.cB1[dp.nI - 1];
        if (dt.fN[ij] > 0)
            dt.h[ij] = -dt.h[ij];

        ij       = dt.idx(1, dp.nJ);
        dt.h[ij] = sqrtf(SQR(dt.fM[ij]) + SQR(dt.fN[dt.dn(ij)])) * dt.cB3[0];
        if (dt.fN[dt.dn(ij)] < 0)
            dt.h[ij] = -dt.h[ij];

        ij       = dt.idx(dp.nI, dp.nJ);
        dt.h[ij] = sqrtf(SQR(dt.fM[dt.le(ij)]) + SQR(dt.fN[dt.dn(ij)])) * dt.cB3[dp.nI - 1];
        if (dt.fN[dt.dn(ij)] < 0)
            dt.h[ij] = -dt.h[ij];
    }
}

__global__ void runFluxBoundaryKernel(KernelData data)
{

    KernelData &dt = data;
    Params     &dp = data.params;

    int id = blockIdx.x * blockDim.x + threadIdx.x + 1;
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

__global__ void runGridExtendKernel(KernelData data)
{

    Params &dp = data.params;

    int id = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (id >= dp.jMin && id <= dp.jMax) {

        if (fabsf(data.h[data.idx(dp.iMin + 2, id)]) > dp.sshClipThreshold)
            atomicAdd(&(data.g_MinMax->x), 1);

        if (fabsf(data.h[data.idx(dp.iMax - 2, id)]) > dp.sshClipThreshold)
            atomicAdd(&(data.g_MinMax->y), 1);
    }

    if (id >= dp.iMin && id <= dp.iMax) {

        if (fabsf(data.h[data.idx(id, dp.jMin + 2)]) > dp.sshClipThreshold)
            atomicAdd(&(data.g_MinMax->z), 1);

        if (fabsf(data.h[data.idx(id, dp.jMax - 2)]) > dp.sshClipThreshold)
            atomicAdd(&(data.g_MinMax->w), 1);
    }
}
