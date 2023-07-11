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

#pragma once

#include <iostream>

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Farm.h>
#include <libethcore/Miner.h>

#include "../PoolClient.h"

using namespace std;
using namespace dev;
using namespace eth;

class SimulateClient : public PoolClient, Worker
{
  public:
    SimulateClient(unsigned const &block);
    ~SimulateClient() override;

    void connect() override;
    void disconnect() override;

    bool   isPendingState() override { return false; }
    string ActiveEndPoint() override { return ""; };

    void submitHashrate(uint64_t const &rate, string const &id) override;
    void submitSolution(const Solution &solution) override;

  private:
    void                                  workLoop() override;
    unsigned                              m_block;
    std::chrono::steady_clock::time_point m_start_time;

    float hr_alpha = 0.45f;
    float hr_max   = 0.0f;
    float hr_mean  = 0.0f;
};
