/* 
 * Copyright (C) <2023> Intel Corporation
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *  
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *  
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 *  
 * 
 * SPDX-License-Identifier: GPL-2.0-or-later
 * 
 */ 

/*
This file is part of ethminer.

ethminer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ethminer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <sycl.hpp>
#include <libethcore/Farm.h>
#include <ethash/ethash.hpp>

#include "SYCLMiner.h"

#include "ethash_sycl_miner_kernel.h"

#include <stdlib.h>

/* Paste this on the file you want to debug. */
#include <stdio.h>
#include <execinfo.h>

#include <iostream>
#include <set>

using namespace dev;
using namespace eth;

struct SYCLChannel : public LogChannel {
    static const char *name() { return EthOrange "sycl"; }
    static const int   verbosity = 2;
};
#define cudalog etherlog(SYCLChannel)

static std::vector<std::string> ExtractLDDPathNameFromProcess(std::vector<std::string> const &vLDDPathsToSearch)
{
    std::string const sProcessMapsFile("/proc/" + std::to_string(::getpid()) + "/maps");
    std::ifstream     inMapsFile(sProcessMapsFile);
    if (!inMapsFile.good()) {
        std::cout << "WARNING: Unable to find process's maps file " << sProcessMapsFile << std::endl;
        return std::vector<std::string>(); // Return empty vector
    }

    std::set<std::string> setUniquePathsFound;
    while (!inMapsFile.eof()) {
        std::string sStringLine("");
        std::getline(inMapsFile, sStringLine);
        if (sStringLine.find_first_of('/') == std::string::npos)
            continue;
        setUniquePathsFound.insert(sStringLine.substr(sStringLine.find_first_of('/'), sStringLine.length()));
    }

    unsigned int const       uiNumberOfPathsToSearch(vLDDPathsToSearch.size());
    std::vector<std::string> vFoundLDDPaths(uiNumberOfPathsToSearch, "");
    for (unsigned int uiPath = 0; uiPath < uiNumberOfPathsToSearch; ++uiPath) {
        for (auto const &sPath : setUniquePathsFound) {
            if (sPath.find(vLDDPathsToSearch[uiPath]) == std::string::npos)
                continue;
            vFoundLDDPaths[uiPath] = sPath;
        }
    }

    return vFoundLDDPaths;
}

static void DisplayDeviceProperties(sycl::device const &Device)
{
    std::cout << std::endl
              << "Device Info:" << std::endl;
    std::cout << "\tUsing SYCL device         : " << Device.get_info<sycl::info::device::name>() << " (Driver version " << Device.get_info<sycl::info::device::driver_version>() << ")" << std::endl;
    std::cout << "\tPlatform                  : " << Device.get_platform().get_info<sycl::info::platform::name>() << std::endl;
    std::cout << "\tVendor                    : " << Device.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "\tMax compute units         : " << Device.get_info<sycl::info::device::max_compute_units>() << std::endl;

    std::vector<std::string> const vLDDPaths(ExtractLDDPathNameFromProcess({"libOpenCL", "libsycl", "libComputeCpp", "libze"})); //[0] OCL, [1] Intel's SYCL, [2] ComputeCpp SYCL
    if (vLDDPaths.empty()) {
        std::cout << "WARNING: Unable to print OpenCL and SYCL dependent libraries! The LD_LIBRARY_PATH may be incorrectly set" << std::endl; // Should not reach to this case
        return;
    }

    std::cout << "\tUsing OpenCL library      : " << (!vLDDPaths[0].empty() ? vLDDPaths[0] : "WARNING! OpenCL library not found!") << std::endl;

    if (!vLDDPaths[1].empty()) { // Implies we are using Intel's DPC++ compiler
        std::cout << "\tUsing OneAPI SYCL library : " << vLDDPaths[1] << std::endl;
        std::cout << "\tUsing Level Zero library  : " << (!vLDDPaths[3].empty() ? vLDDPaths[3] : "WARNING! Level zero library not found! L0 backend may not be available!") << std::endl;
    }

    if (!vLDDPaths[2].empty())
        std::cout << "\tUsing ComputeCPP library  : " << vLDDPaths[2] << std::endl;

    std::cout << "\n\n"
              << std::endl;
}

SYCLMiner::SYCLMiner(unsigned _index, SYSettings _settings, DeviceDescriptor & /*_device */)
    : Miner("sycl-", _index),
      m_settings(_settings),
      m_batch_size(_settings.gridSize * _settings.blockSize),
      m_streams_batch_size(_settings.gridSize * _settings.blockSize * _settings.streams),
      m_DefaultQueue(sycl::queue(sycl::gpu_selector_v))

{
    DisplayDeviceProperties(m_DefaultQueue.get_device());

    try {
        d_dag_size             = sycl::malloc_device<uint32_t>   (1,  m_DefaultQueue);
        d_dag                  = sycl::malloc_device<hash128_t *>(1,  m_DefaultQueue);
        d_light_size           = sycl::malloc_device<uint32_t>   (1,  m_DefaultQueue);
        d_light                = sycl::malloc_device<hash64_t *> (1,  m_DefaultQueue);
        d_header               = sycl::malloc_device<hash32_t>   (1,  m_DefaultQueue);
        d_target               = sycl::malloc_device<uint64_t>   (1,  m_DefaultQueue);
        keccak_round_constants = sycl::malloc_device<sycl::uint2>(24, m_DefaultQueue); // There's 24 constants to store
#ifdef USE_AMD_BACKEND
        pdShuffleOffsets = sycl::malloc_device<int>(64, m_DefaultQueue);
#else
        pdShuffleOffsets = sycl::malloc_device<int>(32, m_DefaultQueue);
#endif
    } catch (sycl::exception const &e) {
        std::cerr << "SYCL exception caught \'" << e.what() << "\'" << std::endl;
        exit(EXIT_FAILURE);
    }
}

SYCLMiner::~SYCLMiner()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cudalog, "sycl-" << m_index << " SYCLMiner::~SYCLMiner() begin");
    stopWorking();
    kick_miner();
    DEV_BUILD_LOG_PROGRAMFLOW(cudalog, "sycl-" << m_index << " SYCLMiner::~SYCLMiner() end");
}

// [JT>>:] This method might be redundant considering that we used a gpu_selector() to select the device
//         Only left in here because it needs information about the device through m_deviceDescriptor
bool SYCLMiner::initDevice()
{
    m_deviceDescriptor.totalMemory = 8000000000; // TODO: Hard coded....
    cudalog << "Using Pci Id : " << m_deviceDescriptor.uniqueId << " " << m_deviceDescriptor.cuName
            << " (Compute " + m_deviceDescriptor.cuCompute + ") Memory : "
            << dev::getFormattedMemory((double)m_deviceDescriptor.totalMemory);

    // Set Hardware Monitor Info
    m_hwmoninfo.deviceType  = HwMonitorInfoType::NVIDIA;
    m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
    m_hwmoninfo.deviceIndex = -1; // Will be later on mapped by nvml (see Farm() constructor)

    cudalog << "Device initialization successfully completed";
    return true;
}

bool SYCLMiner::initEpoch_internal()
{
    // try {
    //  If we get here it means epoch has changed so it's not necessary
    //  to check again dag sizes. They're changed for sure
    bool retVar                = false;
    m_current_target           = 0;
    auto   startInit           = std::chrono::steady_clock::now();
    size_t RequiredTotalMemory = (m_epochContext.dagSize + m_epochContext.lightSize);
    size_t RequiredDagMemory   = m_epochContext.dagSize;

    // Release the pause flag if any
    resume(MinerPauseEnum::PauseDueToInsufficientMemory);
    resume(MinerPauseEnum::PauseDueToInitEpochError);

    bool lightOnHost = false;

    try {
        hash128_t *dag;
        hash64_t  *light;

        // If we have already enough memory allocated, we just have to
        // copy light_cache and regenerate the DAG
        if (m_allocated_memory_dag < m_epochContext.dagSize || m_allocated_memory_light_cache < m_epochContext.lightSize) {

            // Check whether the current device has sufficient memory every time we recreate the dag
            if (m_deviceDescriptor.totalMemory < RequiredTotalMemory) {
                if (m_deviceDescriptor.totalMemory < RequiredDagMemory) {
                    cudalog << "Epoch " << m_epochContext.epochNumber << " requires "
                            << dev::getFormattedMemory((double)RequiredDagMemory) << " memory.";
                    cudalog << "This device hasn't enough memory available. Mining suspended ...";
                    pause(MinerPauseEnum::PauseDueToInsufficientMemory);
                    return true; // This will prevent to exit the thread and
                                 // Eventually resume mining when changing coin or epoch (NiceHash)
                } else
                    lightOnHost = true;
            }

            cudalog << "Generating DAG + Light(on " << (lightOnHost ? "host" : "GPU")
                    << ") : " << dev::getFormattedMemory((double)RequiredTotalMemory);

            // create buffer for cache
            if (lightOnHost) {
                *(reinterpret_cast<void **>(&light)) = (void *)sycl::malloc_host(m_epochContext.lightSize, m_DefaultQueue);
                cudalog << "WARNING: Generating DAG will take minutes, not seconds";
            } else {
                *(reinterpret_cast<void **>(&light)) = (void *)sycl::malloc_device(m_epochContext.lightSize, m_DefaultQueue);
            }

            m_allocated_memory_light_cache     = m_epochContext.lightSize;
            *(reinterpret_cast<void **>(&dag)) = (void *)sycl::malloc_device(m_epochContext.dagSize, m_DefaultQueue);
            m_allocated_memory_dag             = m_epochContext.dagSize;

            // create mining buffers
            cudalog << "Number of queues to create: " << m_settings.streams;
            for (unsigned i = 0; i != m_settings.streams; ++i) {
                m_streams[i] = new sycl::queue(m_DefaultQueue.get_device());
                assert(m_streams[i] != nullptr);
                m_search_buf[i]         = sycl::malloc_device<Search_results>(1, *m_streams[i]);
                m_vHostResults[i].count = 0;
            }
        } else {
            cudalog << "Generating DAG + Light (reusing buffers): " << dev::getFormattedMemory((double)RequiredTotalMemory);
            get_constants(&dag, NULL, &light, NULL);
        }

        m_DefaultQueue.memcpy(reinterpret_cast<void *>(light), m_epochContext.lightCache, m_epochContext.lightSize).wait();

        set_constants(dag, m_epochContext.dagNumItems, light, m_epochContext.lightNumItems);

        ethash_generate_dag(m_epochContext.dagSize, m_settings.gridSize, m_settings.blockSize, m_streams[0]);

        cudalog << "Generated DAG + Light in "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - startInit)
                       .count()
                << " ms. "
                << dev::getFormattedMemory(
                       lightOnHost ? (double)(m_deviceDescriptor.totalMemory - RequiredDagMemory) : (double)(m_deviceDescriptor.totalMemory - RequiredTotalMemory))
                << " left.";

        retVar = true;
    } catch (const cuda_runtime_error &ec) {
        cudalog << "Unexpected error " << ec.what() << " on SYCL device "
                << m_deviceDescriptor.uniqueId;
        cudalog << "Mining suspended ...";
        pause(MinerPauseEnum::PauseDueToInitEpochError);
        retVar = true;
    }

    return retVar;
}

void SYCLMiner::workLoop()
{
    /// try {
    WorkPackage current;
    current.header = h256();

    m_vHostResults.resize(m_settings.streams);
    m_search_buf.resize(m_settings.streams);
    m_streams.resize(m_settings.streams);

    if (!initDevice())
        return;

    try {
        while (!shouldStop()) {
            // Wait for work or 3 seconds (whichever the first)
            const WorkPackage w = work();
            if (!w) {
                boost::system_time const timeout =
                    boost::get_system_time() + boost::posix_time::seconds(3);
                boost::mutex::scoped_lock l(x_work);
                m_new_work_signal.timed_wait(l, timeout);
                continue;
            }

            // Epoch change ?
            if (current.epoch != w.epoch) {
                if (!initEpoch())
                    break; // This will simply exit the thread

                // As DAG generation takes a while we need to
                // ensure we're on latest job, not on the one
                // which triggered the epoch change
                current = w;
                continue;
            }

            // Persist most recent job.
            // Job's differences should be handled at higher level
            current                    = w;
            uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)current.boundary >> 192);

            // Eventually start searching
            search(current.header.data(), upper64OfBoundary, current.startNonce, w);
        }

        // Reset miner and stop working
    } catch (cuda_runtime_error const &_e) {
        string _what = "GPU error: ";
        _what.append(_e.what());
        throw std::runtime_error(_what);
    }
}

void SYCLMiner::kick_miner()
{
    m_new_work.store(true, std::memory_order_relaxed);
    m_new_work_signal.notify_one();
}

void SYCLMiner::enumDevices(std::map<string, DeviceDescriptor> &_DevicesCollection) // [JT>>:] Is this even needed?
{
    try {
        int    i = 0;
        size_t freeMem; ///, totalMem;
        std::vector<sycl::device> vDevices({sycl::device(sycl::gpu_selector_v)});
        for (auto &device : vDevices) {

            std::string const uniqueId = device.get_info<sycl::info::device::name>();

            DeviceDescriptor deviceDescriptor;
            if (_DevicesCollection.find(uniqueId) != _DevicesCollection.end())
                continue;
            else
                deviceDescriptor = DeviceDescriptor();
            std::cout << "Added device " << uniqueId << "\n";
            deviceDescriptor.name              = uniqueId;
            deviceDescriptor.uniqueId          = uniqueId;
            deviceDescriptor.type              = DeviceTypeEnum::Gpu; /// get_device_type(uniqueId);
            deviceDescriptor.syclDetected      = deviceDescriptor.type == DeviceTypeEnum::Gpu ? true : false;
            deviceDescriptor.syclDeviceIndex   = i;
            deviceDescriptor.syclDeviceOrdinal = i;
            deviceDescriptor.syclName          = uniqueId;
            freeMem                            = 8000000000;
            deviceDescriptor.totalMemory       = freeMem;

            _DevicesCollection[uniqueId] = deviceDescriptor;

            i++;
        }
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                  << std::endl;
        std::exit(1);
    }
}

void SYCLMiner::search(uint8_t const *header, uint64_t target, uint64_t start_nonce, const dev::eth::WorkPackage &w)
{
    cudalog << EthWhite << "Launching miner. Global Work Size: " << m_settings.gridSize << " Local Work Size: " << m_settings.blockSize << EthReset;
    try {
        set_header(*reinterpret_cast<hash32_t const *>(header));
        if (m_current_target != target) {
            set_target(target);
            m_current_target = target;
        }

        // prime each stream, clear search result buffers and start the search
        uint32_t current_index;

        for (current_index = 0; current_index < m_settings.streams; current_index++, start_nonce += m_batch_size) {
            sycl::queue    *stream = m_streams[current_index];
            Search_results &buffer(*m_search_buf[current_index]);
            stream->memcpy(&buffer, &m_vHostResults[current_index], sizeof(Search_results)).wait();

            // Run the batch for this stream
            run_ethash_search(m_settings.gridSize, m_settings.blockSize, stream, &buffer, start_nonce);
            m_u64TotalGeneratedHashes += m_settings.gridSize * m_settings.blockSize; // Reached here because hashes were generated
        }

        // process stream batches until we get new work.

        bool done = false;
        while (!done) {
            // Exit next time around if there's new work awaiting
            bool t = true;
            done   = m_new_work.compare_exchange_strong(t, false);

            // Check on every batch if we need to suspend mining
            if (!done)
                done = paused();

            // This inner loop will process each sycl stream individually
            for (current_index = 0; current_index < m_settings.streams; current_index++, start_nonce += m_batch_size) {
                // Each pass of this loop will wait for a stream to exit,
                // save any found solutions, then restart the stream
                // on the next group of nonces.
                sycl::queue *stream = m_streams[current_index];

                // Wait for the stream complete
                stream->wait();

                if (shouldStop()) {
                    m_new_work.store(false, std::memory_order_relaxed);
                    done = true;
                }

                // Detect solutions in current stream's solution buffer
                Search_results &buffer(*m_search_buf[current_index]);
                stream->memcpy(&m_vHostResults[current_index], &buffer, sizeof(Search_results)).wait();

                uint32_t found_count = std::min((unsigned)m_vHostResults[current_index].count, MAX_SEARCH_RESULTS);

                uint32_t gids[MAX_SEARCH_RESULTS];
                h256     mixes[MAX_SEARCH_RESULTS];

                if (found_count) {
                    m_vHostResults[current_index].count = 0; // reset

                    // Extract solution and pass to higer level
                    // using io_service as dispatcher

                    for (uint32_t i = 0; i < found_count; i++) {
                        gids[i] = m_vHostResults[current_index].result[i].gid;
                        memcpy(mixes[i].data(), (void *)&(m_vHostResults[current_index]).result[i].mix, sizeof(m_vHostResults[current_index].result[i].mix));
                    }
                }

                // restart the stream on the next batch of nonces
                // unless we are done for this round.
                if (!done) {
                    stream->memcpy(&buffer, &m_vHostResults[current_index], sizeof(Search_results)).wait();
                    run_ethash_search(m_settings.gridSize, m_settings.blockSize, stream, &buffer, start_nonce);
                    m_u64TotalGeneratedHashes += m_settings.gridSize * m_settings.blockSize; // Reached here because hashes were generated
                }

                if (found_count) {
                    uint64_t nonce_base = start_nonce - m_streams_batch_size;
                    for (uint32_t i = 0; i < found_count; i++) {
                        uint64_t nonce = nonce_base + gids[i];

                        Farm::f().submitProof(Solution{nonce, mixes[i], w, std::chrono::steady_clock::now(), m_index});
                        cudalog << EthWhite << "Job: " << w.header.abridged() << " Sol: 0x" << toHex(nonce) << EthReset;
                    }
                }
            }

            // Update the hash rate
            updateHashRate(m_batch_size, m_settings.streams);

            // Bail out if it's shutdown time
            if (shouldStop()) {
                m_new_work.store(false, std::memory_order_relaxed);
                break;
            }
        }

#ifdef DEV_BUILD
        // Optionally log job switch time
        if (!shouldStop() && (g_logOptions & LOG_SWITCH))
            cudalog << "Switch time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_workSwitchStart).count() << " ms.";
#endif
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }
}
