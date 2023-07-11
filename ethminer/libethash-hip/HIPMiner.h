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

#include "ethash_hip_miner_kernel.h"

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>

#include <functional>

namespace dev
{
namespace eth
{
class HIPMiner : public Miner
{
  public:
    HIPMiner(unsigned _index, HIPSettings _settings, DeviceDescriptor &_device);
    ~HIPMiner() override;

    static int  getNumDevices();
    static void enumDevices(std::map<string, DeviceDescriptor> &_DevicesCollection);

    void search(uint8_t const *header, uint64_t target, uint64_t _startN, const dev::eth::WorkPackage &w);

  protected:
    bool initDevice() override;

    bool initEpoch_internal() override;

    void kick_miner() override;

  private:
    atomic<bool> m_new_work = {false};

    void workLoop() override;

    std::vector<Search_results *> m_search_buf;
    std::vector<hipStream_t>      m_streams;
    uint64_t                      m_current_target = 0;

    std::vector<Search_results> m_vHostResults;

    HIPSettings m_settings;

    const uint32_t m_batch_size;
    const uint32_t m_streams_batch_size;

    uint64_t m_allocated_memory_dag         = 0; // dag_size is a uint64_t in EpochContext struct
    size_t   m_allocated_memory_light_cache = 0;
};

} // namespace eth
} // namespace dev
