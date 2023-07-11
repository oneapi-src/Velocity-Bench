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

#include <libethcore/Farm.h>
#include <ethash/ethash.hpp>

#include "HIPMiner.h"

using namespace std;
using namespace dev;
using namespace eth;

struct HIPChannel : public LogChannel {
    static const char *name() { return EthOrange "hip"; }
    static const int   verbosity = 2;
};
#define hiplog etherlog(HIPChannel)

HIPMiner::HIPMiner(unsigned _index, HIPSettings _settings, DeviceDescriptor &_device)
    : Miner("hip-", _index),
      m_settings(_settings),
      m_batch_size(_settings.gridSize * _settings.blockSize),
      m_streams_batch_size(_settings.gridSize * _settings.blockSize * _settings.streams)
{
    m_deviceDescriptor = _device;
}

HIPMiner::~HIPMiner()
{
    DEV_BUILD_LOG_PROGRAMFLOW(hiplog, "hip-" << m_index << " HIPMiner::~HIPMiner() begin");
    stopWorking();
    kick_miner();
    DEV_BUILD_LOG_PROGRAMFLOW(hiplog, "hip-" << m_index << " HIPMiner::~HIPMiner() end");
}

bool HIPMiner::initDevice()
{
    std::cout << "Initializing device" << std::endl;

    hiplog << "Using Pci Id : " << m_deviceDescriptor.uniqueId << " " << m_deviceDescriptor.cuName
           << " (Compute " + m_deviceDescriptor.cuCompute + ") Memory : "
           << dev::getFormattedMemory((double)m_deviceDescriptor.totalMemory);

    // Set Hardware Monitor Info
    m_hwmoninfo.deviceType  = HwMonitorInfoType::NVIDIA;
    m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
    m_hwmoninfo.deviceIndex = -1; // Will be later on mapped by nvml (see Farm() constructor)

    try {
        CUDA_SAFE_CALL(hipSetDevice(m_deviceDescriptor.cuDeviceIndex));
        CUDA_SAFE_CALL(hipDeviceReset());
    } catch (const cuda_runtime_error &ec) {
        hiplog << "Could not set CUDA device on Pci Id " << m_deviceDescriptor.uniqueId
               << " Error : " << ec.what();
        hiplog << "Mining aborted on this device.";
        return false;
    }
    return true;
}

bool HIPMiner::initEpoch_internal()
{
    // If we get here it means epoch has changed so it's not necessary
    // to check again dag sizes. They're changed for sure
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
        if (m_allocated_memory_dag < m_epochContext.dagSize ||
            m_allocated_memory_light_cache < m_epochContext.lightSize) {
            // We need to reset the device and (re)create the dag
            // hipDeviceReset() frees all previous allocated memory
            CUDA_SAFE_CALL(hipDeviceReset());
            CUDA_SAFE_CALL(hipSetDeviceFlags(m_settings.schedule));
            ////CUDA_SAFE_CALL(hipDeviceSetCacheConfig(hipFuncCachePreferL1));

            // Check whether the current device has sufficient memory every time we recreate the dag
            if (m_deviceDescriptor.totalMemory < RequiredTotalMemory) {
                if (m_deviceDescriptor.totalMemory < RequiredDagMemory) {
                    hiplog << "Epoch " << m_epochContext.epochNumber << " requires "
                           << dev::getFormattedMemory((double)RequiredDagMemory) << " memory.";
                    hiplog << "This device hasn't enough memory available. Mining suspended ...";
                    pause(MinerPauseEnum::PauseDueToInsufficientMemory);
                    return true; // This will prevent to exit the thread and
                                 // Eventually resume mining when changing coin or epoch (NiceHash)
                } else
                    lightOnHost = true;
            }

            hiplog << "Generating DAG + Light(on " << (lightOnHost ? "host" : "GPU")
                   << ") : " << dev::getFormattedMemory((double)RequiredTotalMemory);

            // create buffer for cache
            if (lightOnHost) {
                CUDA_SAFE_CALL(hipHostMalloc(reinterpret_cast<void **>(&light), m_epochContext.lightSize, hipHostMallocDefault));
                hiplog << "WARNING: Generating DAG will take minutes, not seconds";
            } else
                CUDA_SAFE_CALL(
                    hipMalloc(reinterpret_cast<void **>(&light), m_epochContext.lightSize));
            m_allocated_memory_light_cache = m_epochContext.lightSize;
            CUDA_SAFE_CALL(hipMalloc(reinterpret_cast<void **>(&dag), m_epochContext.dagSize));
            m_allocated_memory_dag = m_epochContext.dagSize;

            // create mining buffers
            for (unsigned i = 0; i != m_settings.streams; ++i) {
                CUDA_SAFE_CALL(hipMalloc(&m_search_buf[i], sizeof(Search_results)));
                CUDA_SAFE_CALL(hipStreamCreateWithFlags(&m_streams[i], hipStreamNonBlocking));
                m_vHostResults[i].count = 0;
            }
        } else {
            hiplog << "Generating DAG + Light (reusing buffers): "
                   << dev::getFormattedMemory((double)RequiredTotalMemory);
            get_constants(&dag, NULL, &light, NULL);
        }

        CUDA_SAFE_CALL(hipMemcpy(reinterpret_cast<void *>(light), m_epochContext.lightCache, m_epochContext.lightSize, hipMemcpyHostToDevice));

        set_constants(dag, m_epochContext.dagNumItems, light,
                      m_epochContext.lightNumItems); // in ethash_cuda_miner_kernel.cu

        ethash_generate_dag(
            m_epochContext.dagSize, m_settings.gridSize, m_settings.blockSize, m_streams[0]);

        hiplog << "Generated DAG + Light in "
               << std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - startInit)
                      .count()
               << " ms. "
               << dev::getFormattedMemory(
                      lightOnHost ? (double)(m_deviceDescriptor.totalMemory - RequiredDagMemory) : (double)(m_deviceDescriptor.totalMemory - RequiredTotalMemory))
               << " left.";

        retVar = true;
    } catch (const cuda_runtime_error &ec) {
        hiplog << "Unexpected error " << ec.what() << " on CUDA device "
               << m_deviceDescriptor.uniqueId;
        hiplog << "Mining suspended ...";
        pause(MinerPauseEnum::PauseDueToInitEpochError);
        retVar = true;
    }

    return retVar;
}

void HIPMiner::workLoop()
{
    WorkPackage current;
    current.header = h256();

    m_search_buf.resize(m_settings.streams);
    m_streams.resize(m_settings.streams);
    m_vHostResults.resize(m_settings.streams);

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
        CUDA_SAFE_CALL(hipDeviceReset());
    } catch (cuda_runtime_error const &_e) {
        string _what = "GPU error: ";
        _what.append(_e.what());
        throw std::runtime_error(_what);
    }
}

void HIPMiner::kick_miner()
{
    m_new_work.store(true, std::memory_order_relaxed);
    m_new_work_signal.notify_one();
}

int HIPMiner::getNumDevices()
{
    return 1;
    int        deviceCount;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err == hipSuccess)
        return deviceCount;

    if (err == hipErrorInsufficientDriver) {
        int driverVersion = 0;
        hipDriverGetVersion(&driverVersion);
        if (driverVersion == 0)
            std::cerr << "CUDA Error : No CUDA driver found" << std::endl;
        else
            std::cerr << "CUDA Error : Insufficient CUDA driver " << std::to_string(driverVersion)
                      << std::endl;
    } else {
        std::cerr << "CUDA Error : " << hipGetErrorString(err) << std::endl;
    }

    return 0;
}

void HIPMiner::enumDevices(std::map<string, DeviceDescriptor> &_DevicesCollection)
{
    int numDevices = getNumDevices();

    for (int i = 0; i < numDevices; i++) {
        string           uniqueId;
        ostringstream    s;
        DeviceDescriptor deviceDescriptor;
        hipDeviceProp_t  props;

        try {
            size_t freeMem, totalMem;
            CUDA_SAFE_CALL(hipGetDeviceProperties(&props, i));
            CUDA_SAFE_CALL(hipMemGetInfo(&freeMem, &totalMem));
            s << setw(2) << setfill('0') << hex << props.pciBusID << ":" << setw(2)
              << props.pciDeviceID << ".0";
            uniqueId = s.str();

            if (_DevicesCollection.find(uniqueId) != _DevicesCollection.end())
                deviceDescriptor = _DevicesCollection[uniqueId];
            else
                deviceDescriptor = DeviceDescriptor();

            deviceDescriptor.name             = string(props.name);
            deviceDescriptor.hipDetected      = true;
            deviceDescriptor.uniqueId         = uniqueId;
            deviceDescriptor.type             = DeviceTypeEnum::Gpu;
            deviceDescriptor.hipDeviceIndex   = i;
            deviceDescriptor.hipDeviceOrdinal = i;
            deviceDescriptor.hipName          = string(props.name);
            deviceDescriptor.totalMemory      = freeMem;
            deviceDescriptor.hipCompute =
                (to_string(props.major) + "." + to_string(props.minor));
            deviceDescriptor.hipComputeMajor = props.major;
            deviceDescriptor.hipComputeMinor = props.minor;

            _DevicesCollection[uniqueId] = deviceDescriptor;
        } catch (const cuda_runtime_error &_e) {
            std::cerr << _e.what() << std::endl;
        }
    }
}

void HIPMiner::search(
    uint8_t const               *header,
    uint64_t                     target,
    uint64_t                     start_nonce,
    const dev::eth::WorkPackage &w)
{
    set_header(*reinterpret_cast<hash32_t const *>(header));
    if (m_current_target != target) {
        set_target(target);
        m_current_target = target;
    }

    // prime each stream, clear search result buffers and start the search
    uint32_t current_index;
    for (current_index = 0; current_index < m_settings.streams;
         current_index++, start_nonce += m_batch_size) {
        hipStream_t     stream = m_streams[current_index];
        Search_results &buffer(*m_search_buf[current_index]);
        m_vHostResults[current_index].count = 0;
        CUDA_SAFE_CALL(hipMemcpy(reinterpret_cast<void *>(&buffer), &m_vHostResults[current_index], sizeof(Search_results), hipMemcpyHostToDevice));
        /////buffer.count = 0;

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

        // This inner loop will process each cuda stream individually
        for (current_index = 0; current_index < m_settings.streams;
             current_index++, start_nonce += m_batch_size) {
            // Each pass of this loop will wait for a stream to exit,
            // save any found solutions, then restart the stream
            // on the next group of nonces.
            hipStream_t stream = m_streams[current_index];

            // Wait for the stream complete
            CUDA_SAFE_CALL(hipStreamSynchronize(stream));

            if (shouldStop()) {
                m_new_work.store(false, std::memory_order_relaxed);
                done = true;
            }

            // Detect solutions in current stream's solution buffer
            Search_results &buffer(*m_search_buf[current_index]);
            CUDA_SAFE_CALL(hipMemcpy(reinterpret_cast<void *>(&m_vHostResults[current_index]), &buffer, sizeof(Search_results), hipMemcpyDeviceToHost));

            uint32_t found_count = std::min((unsigned)m_vHostResults[current_index].count, MAX_SEARCH_RESULTS);

            uint32_t gids[MAX_SEARCH_RESULTS];
            h256     mixes[MAX_SEARCH_RESULTS];

            if (found_count) {
                m_vHostResults[current_index].count = 0;

                // Extract solution and pass to higer level
                // using io_service as dispatcher

                for (uint32_t i = 0; i < found_count; i++) {
                    gids[i] = m_vHostResults[current_index].result[i].gid;
                    memcpy(mixes[i].data(), (void *)&m_vHostResults[current_index].result[i].mix, sizeof(m_vHostResults[current_index].result[i].mix));
                }
            }

            // restart the stream on the next batch of nonces
            // unless we are done for this round.
            if (!done) {
                CUDA_SAFE_CALL(hipMemcpy(reinterpret_cast<void *>(&buffer), &m_vHostResults[current_index], sizeof(Search_results), hipMemcpyHostToDevice));
                run_ethash_search(m_settings.gridSize, m_settings.blockSize, stream, &buffer, start_nonce);
                m_u64TotalGeneratedHashes += m_settings.gridSize * m_settings.blockSize; // Reached here because hashes were generated
            }

            if (found_count) {
                uint64_t nonce_base = start_nonce - m_streams_batch_size;
                for (uint32_t i = 0; i < found_count; i++) {
                    uint64_t nonce = nonce_base + gids[i];

                    Farm::f().submitProof(
                        Solution{nonce, mixes[i], w, std::chrono::steady_clock::now(), m_index});
                    hiplog << EthWhite << "Job: " << w.header.abridged() << " Sol: 0x"
                           << toHex(nonce) << EthReset;
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
        hiplog << "Switch time: "
               << std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - m_workSwitchStart)
                      .count()
               << " ms.";
#endif
}
