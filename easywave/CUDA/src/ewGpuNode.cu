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

#define CPP_MODULE "CGPU"
#include "Logging.h"

#include "ewGpuNode.cuh"
#include "ewCudaKernels.cuh"
#include <algorithm>
#include <chrono>
#include <cassert>

CGpuNode::CGpuNode()
{

    pitch  = 0;
    copied = true;

    m_vecTimers.emplace_back(Timer("MemcpyD2H")); // 0
    m_vecTimers.emplace_back(Timer("MemcpyH2D")); // 1
    m_vecTimers.emplace_back(Timer("MemFree"));   // 2
    m_vecTimers.emplace_back(Timer("MemAlloc"));  // 3
    m_vecTimers.emplace_back(Timer("Compute"));   // 4

#ifdef ENABLE_KERNEL_PROFILING
    for (int i = 0; i < 5; i++) {
        cudaEventCreate(&(evtStart[i]));
        cudaEventCreate(&(evtEnd[i]));
        dur[i] = 0.0;
    }
#endif

    pMallocPitch_DoNotUse = nullptr;
}

void CGpuNode::PrintTimingStats()
{
#ifdef ENABLE_GPU_TIMINGS
    LOG("Timing Results | Elapsed time");
    LOG("*****************************");
    LOG("Memory Alloc    | " << m_vecTimers[TIMER_MEMALLOC].GetTimeAsString(Timer::Units::SECONDS));
    LOG("Memory Xfer H2D | " << m_vecTimers[TIMER_MEMH2D].GetTimeAsString(Timer::Units::SECONDS));
    LOG("Memory Xfer D2H | " << m_vecTimers[TIMER_MEMD2H].GetTimeAsString(Timer::Units::SECONDS));
    LOG("Memory free     | " << m_vecTimers[TIMER_MEMFREE].GetTimeAsString(Timer::Units::SECONDS));
    LOG("Compute         | " << m_vecTimers[TIMER_COMPUTE].GetTimeAsString(Timer::Units::SECONDS));
    LOG("Total GPU Time  | " << (m_vecTimers[TIMER_MEMALLOC] + m_vecTimers[TIMER_MEMH2D] + m_vecTimers[TIMER_MEMD2H] + m_vecTimers[TIMER_MEMFREE] + m_vecTimers[TIMER_COMPUTE]).GetTimeAsString(Timer::Units::SECONDS));
    LOG("*****************************");

#ifdef ENABLE_KERNEL_PROFILING
    // TODO: FIX KERNEL TIMERS!!!
    LOG("Kernel time(s)");
    LOG("\t Wave Update  : " << Utility::ConvertTimeToReadable(Node.GetKernelTimeInMilliseconds(KERNEL_WAVE_UPDATE)));
    LOG("\t Wave Boundary: " << Utility::ConvertTimeToReadable(Node.GetKernelTimeInMilliseconds(KERNEL_WAVE_BOUNDARY)));
    LOG("\t Flux Update  : " << Utility::ConvertTimeToReadable(Node.GetKernelTimeInMilliseconds(KERNEL_FLUX_UPDATE)));
    LOG("\t Flux Boundary: " << Utility::ConvertTimeToReadable(Node.GetKernelTimeInMilliseconds(KERNEL_FLUX_BOUNDARY)));
    LOG("\t Grid Extend  : " << Utility::ConvertTimeToReadable(Node.GetKernelTimeInMilliseconds(KERNEL_GRID_EXTEND)));
#endif
#endif
}

int CGpuNode::mallocMem()
{

    LOG("Allocating GPU memory");

    CArrayNode::mallocMem();

    Params &dp = data.params;

    /* fill in some fields here */
    dp.nI                  = NLon;
    dp.nJ                  = NLat;
    dp.sshArrivalThreshold = Par.sshArrivalThreshold;
    dp.sshClipThreshold    = Par.sshClipThreshold;
    dp.sshZeroThreshold    = Par.sshZeroThreshold;
    dp.lpad                = 0;

    size_t nJ_aligned = dp.nJ + dp.lpad;

    // cudaMallocPitch is needed here in order to obtain the pitch value

#ifdef ENABLE_GPU_TIMINGS
    m_vecTimers[TIMER_MEMALLOC].Start();
#endif

    CUDA_CALL(cudaMallocPitch(&(pMallocPitch_DoNotUse), &pitch, nJ_aligned * sizeof(float), dp.nI));
    LOG_ASSERT(pitch != 0, "Failed to compute pitch");
    LOG("Computed pitch in bytes is: " << pitch << ", dp.pI: " << pitch / sizeof(float));

    CUDA_CALL(cudaMalloc(&(data.d), pitch * dp.nI));
    CUDA_CALL(cudaMalloc(&(data.h), pitch * dp.nI));
    CUDA_CALL(cudaMalloc(&(data.hMax), pitch * dp.nI));
    CUDA_CALL(cudaMalloc(&(data.fM), pitch * dp.nI));
    CUDA_CALL(cudaMalloc(&(data.fN), pitch * dp.nI));
    CUDA_CALL(cudaMalloc(&(data.cR1), pitch * dp.nI));
    CUDA_CALL(cudaMalloc(&(data.cR2), pitch * dp.nI));
    CUDA_CALL(cudaMalloc(&(data.cR4), pitch * dp.nI));
    CUDA_CALL(cudaMalloc(&(data.tArr), pitch * dp.nI));
    /* TODO: cR3, cR5 for coriolis */

    /* 1-dim */
    CUDA_CALL(cudaMalloc(&(data.cR6), dp.nJ * sizeof(float)));
    CUDA_CALL(cudaMalloc(&(data.cB1), dp.nI * sizeof(float)));
    CUDA_CALL(cudaMalloc(&(data.cB2), dp.nJ * sizeof(float)));
    CUDA_CALL(cudaMalloc(&(data.cB3), dp.nI * sizeof(float)));
    CUDA_CALL(cudaMalloc(&(data.cB4), dp.nJ * sizeof(float)));

    CUDA_CALL(cudaMalloc(&(data.g_MinMax), sizeof(int4)));

#ifdef ENABLE_GPU_TIMINGS
    m_vecTimers[TIMER_MEMALLOC].Stop();
#endif

    /* TODO: make sure that pitch is a multiple of 4 and the same for each cudaMallocPitch() call */
    dp.pI = pitch / sizeof(float);

    return 0;
}

bool CGpuNode::AlignData(float const *const &pInputData, float *&pOutputAlignedData, int const iNumberOfRows,
                         int const iNumberOfCols,        // From Pitch
                         int const iNumberOfColsPitched) // To Pitch
{
    assert(pInputData != nullptr && pOutputAlignedData != nullptr);
    for (int iRow = 0; iRow < iNumberOfRows; ++iRow) {
        std::move(pInputData + (iRow * iNumberOfCols), pInputData + (iRow * iNumberOfCols) + iNumberOfCols, pOutputAlignedData + (iRow * iNumberOfColsPitched));
    }

    return true;
}

int CGpuNode::copyToGPU()
{

    Params &dp = data.params;

    /* align left grid boundary to a multiple of 32 with an offset 1 */
    Jmin -= (Jmin - 2) % 32;

    /* fill in further fields here */
    dp.iMin = Imin;
    dp.iMax = Imax;
    dp.jMin = Jmin;
    dp.jMax = Jmax;

    d_1D_aligned    = new float[dp.nI * dp.pI];
    h_1D_aligned    = new float[dp.nI * dp.pI];
    hMax_1D_aligned = new float[dp.nI * dp.pI];
    fM_1D_aligned   = new float[dp.nI * dp.pI];
    fN_1D_aligned   = new float[dp.nI * dp.pI];
    cR1_1D_aligned  = new float[dp.nI * dp.pI];
    cR2_1D_aligned  = new float[dp.nI * dp.pI];
    cR4_1D_aligned  = new float[dp.nI * dp.pI];
    tArr_1D_aligned = new float[dp.nI * dp.pI];

    assert(d_1D_aligned != nullptr);
    assert(h_1D_aligned != nullptr);
    assert(hMax_1D_aligned != nullptr);
    assert(fM_1D_aligned != nullptr);
    assert(fN_1D_aligned != nullptr);
    assert(cR1_1D_aligned != nullptr);
    assert(cR2_1D_aligned != nullptr);
    assert(cR4_1D_aligned != nullptr);
    assert(tArr_1D_aligned != nullptr);

    AlignData(d, d_1D_aligned, dp.nI, dp.nJ, dp.pI);
    AlignData(h, h_1D_aligned, dp.nI, dp.nJ, dp.pI);
    AlignData(hMax, hMax_1D_aligned, dp.nI, dp.nJ, dp.pI);
    AlignData(fM, fM_1D_aligned, dp.nI, dp.nJ, dp.pI);
    AlignData(fN, fN_1D_aligned, dp.nI, dp.nJ, dp.pI);
    AlignData(cR1, cR1_1D_aligned, dp.nI, dp.nJ, dp.pI);
    AlignData(cR2, cR2_1D_aligned, dp.nI, dp.nJ, dp.pI);
    AlignData(cR4, cR4_1D_aligned, dp.nI, dp.nJ, dp.pI);
    AlignData(tArr, tArr_1D_aligned, dp.nI, dp.nJ, dp.pI);

#ifdef ENABLE_GPU_TIMINGS
    m_vecTimers[TIMER_MEMH2D].Start();
#endif

    /* add offset to data.d to guarantee alignment: data.d + LPAD */
    /* 2-dim */

    CUDA_CALL(cudaMemcpy(data.d, d_1D_aligned, dp.nI * dp.pI * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.h, h_1D_aligned, dp.nI * dp.pI * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.hMax, hMax_1D_aligned, dp.nI * dp.pI * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.fM, fM_1D_aligned, dp.nI * dp.pI * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.fN, fN_1D_aligned, dp.nI * dp.pI * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.cR1, cR1_1D_aligned, dp.nI * dp.pI * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.cR2, cR2_1D_aligned, dp.nI * dp.pI * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.cR4, cR4_1D_aligned, dp.nI * dp.pI * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.tArr, tArr_1D_aligned, dp.nI * dp.pI * sizeof(float), cudaMemcpyHostToDevice));

    /* FIXME: move global variables into data structure */
    /* 1-dim */
    CUDA_CALL(cudaMemcpy(data.cR6, R6, dp.nJ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.cB1, C1, dp.nI * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.cB2, C2, dp.nJ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.cB3, C3, dp.nI * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data.cB4, C4, dp.nJ * sizeof(float), cudaMemcpyHostToDevice));

#ifdef ENABLE_GPU_TIMINGS
    m_vecTimers[TIMER_MEMH2D].Stop();

    LOG("Data copy to GPU completed, Time elapsed: " << m_vecTimers[TIMER_MEMH2D].GetTimeAsString(Timer::Units::SECONDS));
#endif

    return 0;
}
int CGpuNode::copyFromGPU()
{

    Params &dp = data.params;
#ifdef ENABLE_GPU_TIMINGS
    std::chrono::steady_clock::time_point const tStart(std::chrono::steady_clock::now());
#endif
    CUDA_CALL(cudaMemcpy(hMax_1D_aligned, data.hMax, dp.nI * dp.pI * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tArr_1D_aligned, data.tArr, dp.nI * dp.pI * sizeof(float), cudaMemcpyDeviceToHost));
#ifdef ENABLE_GPU_TIMINGS
    std::chrono::steady_clock::time_point const tStop(std::chrono::steady_clock::now());
    m_vecTimers[TIMER_MEMD2H] += std::chrono::steady_clock::duration(tStop - tStart);
#endif
    AlignData(hMax_1D_aligned, hMax, dp.nI, dp.pI, dp.nJ);
    AlignData(tArr_1D_aligned, tArr, dp.nI, dp.pI, dp.nJ);

    return 0;
}

int CGpuNode::copyIntermediate()
{

    /* ignore copy requests if data already present on CPU side */
    if (copied)
        return 0;

    Params &dp = data.params;
#ifdef ENABLE_GPU_TIMINGS
    std::chrono::steady_clock::time_point const tStart(std::chrono::steady_clock::now());
#endif
    CUDA_CALL(cudaMemcpy(h_1D_aligned, data.h, dp.nI * dp.pI * sizeof(float), cudaMemcpyDeviceToHost));
#ifdef ENABLE_GPU_TIMINGS
    std::chrono::steady_clock::time_point const tStop(std::chrono::steady_clock::now());
    m_vecTimers[TIMER_MEMD2H] += std::chrono::steady_clock::duration(tStop - tStart);
#endif
    AlignData(h_1D_aligned, h, dp.nI, dp.pI, dp.nJ);
    /* copy finished */
    copied = true;

    return 0;
}

int CGpuNode::copyPOIs()
{

    LOG_WARNING("Copying POIs, this may prolong total time");
    Params &dp = data.params;

    if (copied)
        return 0;

    for (int n = 0; n < NPOIs; n++) {

        int i = idxPOI[n] / dp.nJ + 1;
        int j = idxPOI[n] % dp.nJ + 1;

        int id = data.idx(i, j);
#ifdef ENABLE_GPU_TIMINGS
        std::chrono::steady_clock::time_point const tStart(std::chrono::steady_clock::now());
#endif
        CUDA_CALL(cudaMemcpy(h + idxPOI[n], data.h + id, sizeof(float), cudaMemcpyDeviceToHost));
#ifdef ENABLE_GPU_TIMINGS
        std::chrono::steady_clock::time_point const tStop(std::chrono::steady_clock::now());
        m_vecTimers[TIMER_MEMD2H] += std::chrono::steady_clock::duration(tStop - tStart);
#endif
    }

    return 0;
}

int CGpuNode::freeMem()
{

    /* 2-dim */
#ifdef ENABLE_GPU_TIMINGS
    m_vecTimers[TIMER_MEMFREE].Start();
#endif
    CUDA_CALL(cudaFree(data.d));
    CUDA_CALL(cudaFree(data.h));
    CUDA_CALL(cudaFree(data.hMax));
    CUDA_CALL(cudaFree(data.fM));
    CUDA_CALL(cudaFree(data.fN));
    CUDA_CALL(cudaFree(data.cR1));
    CUDA_CALL(cudaFree(data.cR2));
    CUDA_CALL(cudaFree(data.cR4));
    CUDA_CALL(cudaFree(data.tArr));

    /* 1-dim */
    CUDA_CALL(cudaFree(data.cR6));
    CUDA_CALL(cudaFree(data.cB1));
    CUDA_CALL(cudaFree(data.cB2));
    CUDA_CALL(cudaFree(data.cB3));
    CUDA_CALL(cudaFree(data.cB4));

    CUDA_CALL(cudaFree(data.g_MinMax));
    CUDA_CALL(cudaFree(pMallocPitch_DoNotUse));
#ifdef ENABLE_GPU_TIMINGS
    m_vecTimers[TIMER_MEMFREE].Stop();
#endif

#ifdef ENABLE_KERNEL_PROFILING
    float total_dur = 0.f;
    for (int j = 0; j < 5; j++) {
        printf_v("Duration %u: %.3f\n", j, dur[j]);
        total_dur += dur[j];
    }
    printf_v("Duration total: %.3f\n", total_dur);
#endif

    CArrayNode::freeMem();

    return 0;
}

int CGpuNode::run()
{

    Params &dp = data.params;

    int nThreads = 256;
    int xThreads = 32;
    int yThreads = nThreads / xThreads;

    int NJ      = dp.jMax - dp.jMin + 1;
    int NI      = dp.iMax - dp.iMin + 1;
    int xBlocks = ceil((float)NJ / (float)xThreads);
    int yBlocks = ceil((float)NI / (float)yThreads);

    dim3 threads(xThreads, yThreads);
    dim3 blocks(xBlocks, yBlocks);

    int nBlocks = ceil((float)std::max(dp.nI, dp.nJ) / (float)nThreads);

    dp.mTime = Par.time;

#ifdef ENABLE_KERNEL_PROFILING
    CUDA_CALL(cudaEventRecord(evtStart[0], 0));
    runWaveUpdateKernel<<<blocks, threads>>>(data);
    CUDA_CALL(cudaEventRecord(evtEnd[0], 0));
    CUDA_CALL(cudaEventRecord(evtStart[1], 0));
    runWaveBoundaryKernel<<<nBlocks, nThreads>>>(data);
    CUDA_CALL(cudaEventRecord(evtEnd[1], 0));
    CUDA_CALL(cudaEventRecord(evtStart[2], 0));
    runFluxUpdateKernel<<<blocks, threads>>>(data);
    CUDA_CALL(cudaEventRecord(evtEnd[2], 0));
    CUDA_CALL(cudaEventRecord(evtStart[3], 0));
    runFluxBoundaryKernel<<<nBlocks, nThreads>>>(data);
    CUDA_CALL(cudaEventRecord(evtEnd[3], 0));
    CUDA_CALL(cudaEventRecord(evtStart[4], 0));
    CUDA_CALL(cudaMemset(data.g_MinMax, 0, sizeof(int4)));
    runGridExtendKernel<<<nBlocks, nThreads>>>(data);
    CUDA_CALL(cudaEventRecord(evtEnd[4], 0));
#else
#ifdef ENABLE_GPU_TIMINGS
    std::chrono::steady_clock::time_point const tStart(std::chrono::steady_clock::now());
#endif
    runWaveUpdateKernel<<<blocks, threads>>>(data);
    checkLastCUDAError();
    runWaveBoundaryKernel<<<nBlocks, nThreads>>>(data);
    checkLastCUDAError();
    runFluxUpdateKernel<<<blocks, threads>>>(data);
    checkLastCUDAError();
    runFluxBoundaryKernel<<<nBlocks, nThreads>>>(data);
    checkLastCUDAError();
    CUDA_CALL(cudaMemset(data.g_MinMax, 0, sizeof(int4)));
    runGridExtendKernel<<<nBlocks, nThreads>>>(data);
    checkLastCUDAError();
    cudaDeviceSynchronize();
    checkLastCUDAError();

#ifdef ENABLE_GPU_TIMINGS
    std::chrono::steady_clock::time_point const tStop(std::chrono::steady_clock::now());

    m_vecTimers[TIMER_COMPUTE] += std::chrono::steady_clock::duration(tStop - tStart);
#endif

#endif

    int4 MinMax;
#ifdef ENABLE_GPU_TIMINGS
    std::chrono::steady_clock::time_point const tStartD2H(std::chrono::steady_clock::now());
#endif
    CUDA_CALL(cudaMemcpy(&MinMax, data.g_MinMax, sizeof(int4), cudaMemcpyDeviceToHost));
#ifdef ENABLE_GPU_TIMINGS
    std::chrono::steady_clock::time_point const tStopD2H(std::chrono::steady_clock::now());
    m_vecTimers[TIMER_MEMD2H] += std::chrono::duration(tStopD2H - tStartD2H);
#endif

    if (MinMax.x)
        Imin = dp.iMin = std::max(dp.iMin - 1, 2);

    if (MinMax.y)
        Imax = dp.iMax = std::min(dp.iMax + 1, dp.nI - 1);

    if (MinMax.z)
        Jmin = dp.jMin = std::max(dp.jMin - 32, 2);

    if (MinMax.w)
        Jmax = dp.jMax = std::min(dp.jMax + 1, dp.nJ - 1);

#ifdef ENABLE_KERNEL_PROFILING
    float _dur;
    for (int j = 0; j < 5; j++) {
        cudaEventElapsedTime(&_dur, evtStart[j], evtEnd[j]);
        dur[j] += _dur;
    }
#endif

    /* data has changed now -> copy becomes necessary */
    copied = false;

    return 0;
}
