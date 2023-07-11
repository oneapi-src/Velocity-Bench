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

#include <sycl.hpp>
#include "ewCudaKernels.hpp"
#include "ewGpuNode.hpp"
#include <cmath>

#include <algorithm>

#include <chrono>

#define INT_CEIL(x, n) ((((x) + (n)-1) / (n)) * (n))

// PITCH function was incorporated from dpct header, this is needed
// for 2D memory copy

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))

CGpuNode::CGpuNode() : m_syclHandler(nullptr)
{

    pitch  = 0;
    copied = true;

    for (int i = 0; i < 5; i++) {
        dur[i] = 0.0;
    }
#ifdef ENABLE_GPU_TIMINGS
    m_vecTimers.emplace_back(Timer("MemcpyD2H"));  // 0
    m_vecTimers.emplace_back(Timer("MemcpyH2D"));  // 1
    m_vecTimers.emplace_back(Timer("MemFree"));    // 2
    m_vecTimers.emplace_back(Timer("MemAlloc"));   // 3
    m_vecTimers.emplace_back(Timer("Compute"));    // 4
    m_vecTimers.emplace_back(Timer("Queue_Ctor")); // 5
#endif

    sycl::device const SelectedDevice(sycl::default_selector_v); // Separated device selection out of the queue
#ifdef ENABLE_GPU_TIMINGS
    m_vecTimers[TIMER_QUEUE].Start();
#endif
    m_syclHandler = std::unique_ptr<SYCL>(new SYCL(SelectedDevice, true));
#ifdef ENABLE_GPU_TIMINGS
    m_vecTimers[TIMER_QUEUE].Stop();
#endif
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
    LOG("Queue C-Tor     | " << m_vecTimers[TIMER_QUEUE].GetTimeAsString(Timer::Units::SECONDS));
    LOG("Compute         | " << m_vecTimers[TIMER_COMPUTE].GetTimeAsString(Timer::Units::SECONDS));
    LOG("Total GPU Time  | " << (m_vecTimers[TIMER_MEMALLOC] + m_vecTimers[TIMER_MEMH2D] + m_vecTimers[TIMER_MEMD2H] + m_vecTimers[TIMER_MEMFREE] + m_vecTimers[TIMER_COMPUTE] + m_vecTimers[TIMER_QUEUE]).GetTimeAsString(Timer::Units::SECONDS));
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
    try {
        LOG("Allocating GPU memory");

        CArrayNode::mallocMem();

        Params &dp = data.params;

        /* fill in some fields here */
        dp.nI                  = NLon;
        dp.nJ                  = NLat;
        dp.sshArrivalThreshold = Par.sshArrivalThreshold;
        dp.sshClipThreshold    = Par.sshClipThreshold;
        dp.sshZeroThreshold    = Par.sshZeroThreshold;
        dp.lpad                = 0; // Is this needed?

        size_t nJ_aligned = dp.nJ;
#ifdef ENABLE_GPU_TIMINGS
        m_vecTimers[TIMER_MEMALLOC].Start();
#endif
        pitch = PITCH_DEFAULT_ALIGN(nJ_aligned * sizeof(float));
        LOG_ASSERT(pitch != 0, "Pitch should not be zero!");
        unsigned int const uiNumberOfAlignedBytesToAllocate(pitch * dp.nI); // pitch (in bytes) per row * number of aligned rows * height of the matrix

        /* 2-dim */
        data.d    = reinterpret_cast<float *>(sycl::malloc_device(uiNumberOfAlignedBytesToAllocate, m_syclHandler->GetQueue()));
        data.h    = reinterpret_cast<float *>(sycl::malloc_device(uiNumberOfAlignedBytesToAllocate, m_syclHandler->GetQueue()));
        data.hMax = reinterpret_cast<float *>(sycl::malloc_device(uiNumberOfAlignedBytesToAllocate, m_syclHandler->GetQueue()));
        data.fM   = reinterpret_cast<float *>(sycl::malloc_device(uiNumberOfAlignedBytesToAllocate, m_syclHandler->GetQueue()));
        data.fN   = reinterpret_cast<float *>(sycl::malloc_device(uiNumberOfAlignedBytesToAllocate, m_syclHandler->GetQueue()));
        data.cR1  = reinterpret_cast<float *>(sycl::malloc_device(uiNumberOfAlignedBytesToAllocate, m_syclHandler->GetQueue()));
        data.cR2  = reinterpret_cast<float *>(sycl::malloc_device(uiNumberOfAlignedBytesToAllocate, m_syclHandler->GetQueue()));
        data.cR4  = reinterpret_cast<float *>(sycl::malloc_device(uiNumberOfAlignedBytesToAllocate, m_syclHandler->GetQueue()));
        data.tArr = reinterpret_cast<float *>(sycl::malloc_device(uiNumberOfAlignedBytesToAllocate, m_syclHandler->GetQueue()));
        /* TODO: cR3, cR5 for coriolis */

        /* 1-dim */
        data.cR6      = sycl::malloc_device<float>(dp.nJ, m_syclHandler->GetQueue());
        data.cB1      = sycl::malloc_device<float>(dp.nI, m_syclHandler->GetQueue());
        data.cB2      = sycl::malloc_device<float>(dp.nJ, m_syclHandler->GetQueue());
        data.cB3      = sycl::malloc_device<float>(dp.nI, m_syclHandler->GetQueue());
        data.cB4      = sycl::malloc_device<float>(dp.nJ, m_syclHandler->GetQueue());
        data.g_MinMax = sycl::malloc_device<sycl::int4>(1, m_syclHandler->GetQueue());
#ifdef ENABLE_GPU_TIMINGS
        m_vecTimers[TIMER_MEMALLOC].Stop();
#endif
        /* TODO: make sure that pitch is a multiple of 4 and the same for each cudaMallocPitch() call */
        dp.pI = pitch / sizeof(float);

        return 0;
    } catch (sycl::exception const &e) {
        LOG_ERROR("SYCL exception caught \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception was caught ...");
    }
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
    try {
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

        m_syclHandler->GetQueue().memcpy(data.d, d_1D_aligned, dp.nI * dp.pI * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.h, h_1D_aligned, dp.nI * dp.pI * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.hMax, hMax_1D_aligned, dp.nI * dp.pI * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.fM, fM_1D_aligned, dp.nI * dp.pI * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.fN, fN_1D_aligned, dp.nI * dp.pI * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.cR1, cR1_1D_aligned, dp.nI * dp.pI * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.cR2, cR2_1D_aligned, dp.nI * dp.pI * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.cR4, cR4_1D_aligned, dp.nI * dp.pI * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.tArr, tArr_1D_aligned, dp.nI * dp.pI * sizeof(float)).wait();

        /* FIXME: move global variables into data structure */
        /* 1-dim */
        m_syclHandler->GetQueue().memcpy(data.cR6, R6, dp.nJ * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.cB1, C1, dp.nI * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.cB2, C2, dp.nJ * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.cB3, C3, dp.nI * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(data.cB4, C4, dp.nJ * sizeof(float)).wait();
#ifdef ENABLE_GPU_TIMINGS
        m_vecTimers[TIMER_MEMH2D].Stop();
        LOG("Data copy to GPU completed, Time elapsed: " << m_vecTimers[TIMER_MEMH2D].GetTimeAsString(Timer::Units::SECONDS));
#endif

        return 0;
    } catch (sycl::exception const &e) {
        LOG_ERROR("SYCL exception caught \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception was caught ...");
    }
}

int CGpuNode::copyFromGPU()
{
    try {
        Params &dp = data.params;

#ifdef ENABLE_GPU_TIMINGS
        std::chrono::steady_clock::time_point const tStart(std::chrono::steady_clock::now());
#endif
        m_syclHandler->GetQueue().memcpy(hMax_1D_aligned, data.hMax, dp.nI * dp.pI * sizeof(float)).wait();
        m_syclHandler->GetQueue().memcpy(tArr_1D_aligned, data.tArr, dp.nI * dp.pI * sizeof(float)).wait();
#ifdef ENABLE_GPU_TIMINGS
        std::chrono::steady_clock::time_point const tStop(std::chrono::steady_clock::now());
        m_vecTimers[TIMER_MEMD2H] += std::chrono::duration(tStop - tStart);
#endif
        AlignData(hMax_1D_aligned, hMax, dp.nI, dp.pI, dp.nJ);
        AlignData(tArr_1D_aligned, tArr, dp.nI, dp.pI, dp.nJ);

        return 0;
    } catch (sycl::exception const &e) {
        LOG_ERROR("SYCL exception caught \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception was caught ...");
    }
}

int CGpuNode::copyIntermediate()
{
    try {
        if (copied)
            return 0;

        Params &dp = data.params;
#ifdef ENABLE_GPU_TIMINGS
        std::chrono::steady_clock::time_point const tStart(std::chrono::steady_clock::now());
#endif
        m_syclHandler->GetQueue().memcpy(h_1D_aligned, data.h, dp.nI * dp.pI * sizeof(float)).wait();
#ifdef ENABLE_GPU_TIMINGS
        std::chrono::steady_clock::time_point const tStop(std::chrono::steady_clock::now());
        m_vecTimers[TIMER_MEMD2H] += std::chrono::duration(tStop - tStart);
#endif
        AlignData(h_1D_aligned, h, dp.nI, dp.pI, dp.nJ);

        /* copy finished */
        copied = true;

        return 0;
    } catch (sycl::exception const &e) {
        LOG_ERROR("SYCL exception caught \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception was caught ...");
    }
}

int CGpuNode::copyPOIs()
{
    try {

        LOG_WARNING("Copying POIs, this may prolong total time");
        Params &dp = data.params;

        if (copied)
            return 0;

        for (int n = 0; n < NPOIs; n++) {

            int i = idxPOI[n] / dp.nJ + 1;
            int j = idxPOI[n] % dp.nJ + 1;

            int id = data.idx(i, j);

            std::chrono::steady_clock::time_point const tStart(std::chrono::steady_clock::now());
            m_syclHandler->GetQueue().memcpy(h + idxPOI[n], data.h + id, sizeof(float)).wait();
            std::chrono::steady_clock::time_point const tStop(std::chrono::steady_clock::now());
            m_vecTimers[TIMER_MEMD2H] += std::chrono::steady_clock::duration(tStop - tStart);
        }

        return 0;
    } catch (sycl::exception const &e) {
        LOG_ERROR("SYCL exception caught \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception was caught ...");
    }
}

int CGpuNode::freeMem()
{
    try {
#ifdef ENABLE_GPU_TIMINGS
        m_vecTimers[TIMER_MEMFREE].Start();
#endif
        sycl::free(data.d, m_syclHandler->GetQueue());
        sycl::free(data.h, m_syclHandler->GetQueue());
        sycl::free(data.hMax, m_syclHandler->GetQueue());
        sycl::free(data.fM, m_syclHandler->GetQueue());
        sycl::free(data.fN, m_syclHandler->GetQueue());
        sycl::free(data.cR1, m_syclHandler->GetQueue());
        sycl::free(data.cR2, m_syclHandler->GetQueue());
        sycl::free(data.cR4, m_syclHandler->GetQueue());
        sycl::free(data.tArr, m_syclHandler->GetQueue());

        /* 1-dim */
        sycl::free(data.cR6, m_syclHandler->GetQueue());
        sycl::free(data.cB1, m_syclHandler->GetQueue());
        sycl::free(data.cB2, m_syclHandler->GetQueue());
        sycl::free(data.cB3, m_syclHandler->GetQueue());
        sycl::free(data.cB4, m_syclHandler->GetQueue());

        sycl::free(data.g_MinMax, m_syclHandler->GetQueue());
#ifdef ENABLE_GPU_TIMINGS
        m_vecTimers[TIMER_MEMFREE].Stop();
#endif

        float total_dur = 0.f;
        for (int j = 0; j < 5; j++) {
            printf_v("Duration %u: %.3f\n", j, dur[j]);
            total_dur += dur[j];
        }
        printf_v("Duration total: %.3f\n", total_dur);

        CArrayNode::freeMem();

        return 0;
    } catch (sycl::exception const &e) {
        LOG_ERROR("SYCL exception caught \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception was caught ...");
    }
}

int CGpuNode::run()
{
    try {

        Params &dp = data.params;

        int NJ = dp.jMax - dp.jMin + 1;
        int NI = dp.iMax - dp.iMin + 1;

        //size_t max_wg_size = m_syclHandler->GetQueue().get_device().get_info<sycl::info::device::max_work_group_size>();

        //sycl::range<1> boundary_workgroup_size(max_wg_size);
	sycl::range<1> boundary_workgroup_size(256);
        sycl::range<1> boundary_size(INT_CEIL(std::max(dp.nI, dp.nJ), boundary_workgroup_size[0]));

#if defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
        /* For Intel, prevent the nd_range_error: "Non-uniform work-groups are not supported by the target device -54 (CL_INVALID_WORK_GROUP_SIZE))". */
        /* Originally we had n = 128 threads, 32 for x and 128/x = 4 threads, hardcoded in the CUDA code. */
        sycl::range<2> compute_wnd_workgroup_size(4, 32);
#else
        sycl::range<2> compute_wnd_workgroup_size(32, 32);
#endif
        sycl::range<2> compute_wnd_size(
            INT_CEIL(NI, compute_wnd_workgroup_size[0]),
            INT_CEIL(NJ, compute_wnd_workgroup_size[1]));

        dp.mTime = Par.time;
#ifdef ENABLE_GPU_TIMINGS
        std::chrono::steady_clock::time_point const tStart(std::chrono::steady_clock::now());
#endif
        m_syclHandler->GetQueue().submit([&](sycl::handler &cgh) {
            auto kernel_data = data;

            cgh.parallel_for<class runWaveUpdate>(
                sycl::nd_range<2>(compute_wnd_size, compute_wnd_workgroup_size),
                [=](sycl::nd_item<2> item) {
                    waveUpdate(kernel_data, item);
                });
        });

        m_syclHandler->GetQueue().submit([&](sycl::handler &cgh) {
            auto kernel_data = data;
            cgh.parallel_for<class runWaveBoundary>(
                sycl::nd_range<1>(boundary_size, boundary_workgroup_size),
                [=](sycl::nd_item<1> item) {
                    waveBoundary(kernel_data, item);
                });
        });

        m_syclHandler->GetQueue().submit([&](sycl::handler &cgh) {
            auto kernel_data = data;
            cgh.parallel_for<class runFluxUpdate>(
                sycl::nd_range<2>(compute_wnd_size, compute_wnd_workgroup_size),
                [=](sycl::nd_item<2> item) {
                    fluxUpdate(kernel_data, item);
                });
        });

        m_syclHandler->GetQueue().submit([&](sycl::handler &cgh) {
            auto kernel_data = data;
            cgh.parallel_for<class runFluxBoundary>(sycl::nd_range<1>(boundary_size, boundary_workgroup_size), [=](sycl::nd_item<1> item) {
                fluxBoundary(kernel_data, item);
            });
        });

        m_syclHandler->GetQueue().memset(data.g_MinMax, 0, sizeof(sycl::int4));

        m_syclHandler->GetQueue().submit([&](sycl::handler &cgh) {
            auto kernel_data = data;
            cgh.parallel_for<class runGridExtend>(sycl::nd_range<1>(boundary_size, boundary_workgroup_size), [=](sycl::nd_item<1> item) {
                gridExtend(kernel_data, item);
            });
        });

        m_syclHandler->GetQueue().wait_and_throw();
#ifdef ENABLE_GPU_TIMINGS
        std::chrono::steady_clock::time_point const tStop(std::chrono::steady_clock::now());
        m_vecTimers[TIMER_COMPUTE] += std::chrono::duration(tStop - tStart);
#endif
        sycl::int4 MinMax;

        std::chrono::steady_clock::time_point const tStartD2H(std::chrono::steady_clock::now());
        m_syclHandler->GetQueue().memcpy(&MinMax, data.g_MinMax, sizeof(sycl::int4)).wait();
        std::chrono::steady_clock::time_point const tStopD2H(std::chrono::steady_clock::now());
#ifdef ENABLE_GPU_TIMINGS
        m_vecTimers[TIMER_MEMD2H] += std::chrono::duration(tStopD2H - tStartD2H);
#endif

        /* TODO: respect alignments from device in window expansion (Preferred work group size multiple ?!) */
        if (MinMax.x())
            Imin = dp.iMin = std::max(dp.iMin - 1, 2);
        if (MinMax.y())
            Imax = dp.iMax = std::min(dp.iMax + 1, dp.nI - 1);
        if (MinMax.z())
            Jmin = dp.jMin = std::max(dp.jMin - MEM_ALIGN, 2);
        if (MinMax.w())
            Jmax = dp.jMax = std::min(dp.jMax + 1, dp.nJ - 1);

        /* data has changed now -> copy becomes necessary */
        copied = false;

        return 0;

    } catch (sycl::exception const &e) {
        LOG_ERROR("SYCL exception caught \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception was caught ...");
    }
}
