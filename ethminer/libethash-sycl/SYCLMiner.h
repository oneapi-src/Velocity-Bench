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

#pragma once

#include <sycl/sycl.hpp>
#include "ethash_sycl_miner_kernel.h"

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>

#include <functional>

namespace dev
{
namespace eth
{
class SYCLMiner : public Miner
{
  public:
    SYCLMiner(unsigned _index, SYSettings _settings, DeviceDescriptor &_device);
    ~SYCLMiner() override;

    static int  getNumDevices();
    static void enumDevices(std::map<string, DeviceDescriptor> &_DevicesCollection);

    void search(
        uint8_t const               *header,
        uint64_t                     target,
        uint64_t                     _startN,
        const dev::eth::WorkPackage &w);

  protected:
    bool initDevice() override;

    bool initEpoch_internal() override;

    void kick_miner() override;

  private:
    uint32_t    *d_dag_size;
    hash128_t  **d_dag;
    uint32_t    *d_light_size;
    hash64_t   **d_light;
    hash32_t    *d_header;
    uint64_t    *d_target;
    sycl::uint2 *keccak_round_constants;
    int         *pdShuffleOffsets;

    atomic<bool> m_new_work = {false};

    void workLoop() override;

    std::vector<Search_results>   m_vHostResults;
    std::vector<Search_results *> m_search_buf;

    std::vector<sycl::queue *> m_streams;

    uint64_t m_current_target = 0;

    SYSettings m_settings;

    const uint32_t m_batch_size;
    const uint32_t m_streams_batch_size;
    sycl::queue    m_DefaultQueue;

    uint64_t m_allocated_memory_dag         = 0; // dag_size is a uint64_t in EpochContext struct
    size_t   m_allocated_memory_light_cache = 0;

    void set_constants(hash128_t *_dag, uint32_t _dag_size, hash64_t *_light, uint32_t _light_size);
    void get_constants(hash128_t **_dag, uint32_t *_dag_size, hash64_t **_light, uint32_t *_light_size);

    void set_header(hash32_t _header);

    void set_target(uint64_t _target);

    void run_ethash_search(uint32_t gridSize, uint32_t blockSize, sycl::queue *stream, Search_results *g_output, uint64_t start_nonce);

    void ethash_generate_dag(uint64_t dag_size, uint32_t blocks, uint32_t threads, sycl::queue *stream);
};

} // namespace eth
} // namespace dev
