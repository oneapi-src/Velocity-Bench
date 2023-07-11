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

#include <json/json.h>

#include <libdevcore/Worker.h>
#include <libethcore/Farm.h>
#include <libethcore/Miner.h>

#include "PoolClient.h"
#include "getwork/EthGetworkClient.h"
#include "stratum/EthStratumClient.h"
#include "testing/SimulateClient.h"

using namespace std;

namespace dev
{
namespace eth
{
struct PoolSettings {
    std::vector<std::shared_ptr<URI>> connections;                 // List of connection definitions
    unsigned                          getWorkPollInterval = 500;   // Interval (ms) between getwork requests
    unsigned                          noWorkTimeout       = 180;   // If no new jobs in this number of seconds drop connection
    unsigned                          noResponseTimeout   = 2;     // If no response in this number of seconds drop connection
    unsigned                          poolFailoverTimeout = 0;     // Return to primary pool after this number of minutes
    bool                              reportHashrate      = false; // Whether or not to report hashrate to pool
    unsigned                          hashRateInterval    = 60;    // Interval in seconds among hashrate submissions
    std::string                       hashRateId =
        h256::random().hex(HexPrefix::Add); // Unique identifier for HashRate submission
    unsigned connectionMaxRetries = 3;      // Max number of connection retries
    unsigned delayBeforeRetry     = 0;      // Delay seconds before connect retry
    unsigned benchmarkBlock       = 0;      // Block number used by SimulateClient to test performances
};

class PoolManager
{
  public:
    PoolManager(PoolSettings _settings);
    static PoolManager  &p() { return *m_this; }
    void                 addConnection(std::string _connstring);
    void                 addConnection(std::shared_ptr<URI> _uri);
    Json::Value          getConnectionsJson();
    void                 setActiveConnection(unsigned int idx);
    void                 setActiveConnection(std::string &_connstring);
    std::shared_ptr<URI> getActiveConnection();
    void                 removeConnection(unsigned int idx);
    void                 start();
    void                 stop();
    bool                 isConnected() { return p_client->isConnected(); };
    bool                 isRunning() { return m_running; };
    int                  getCurrentEpoch();
    double               getCurrentDifficulty();
    unsigned             getConnectionSwitches();
    unsigned             getEpochChanges();
    uint64_t             GetTotalGeneratedHashes() { return m_u64TotalNumberOfHashes; }

  private:
    void rotateConnect();

    void setClientHandlers();

    void showMiningAt();

    void setActiveConnectionCommon(unsigned int idx);

    PoolSettings m_Settings;

    void failovertimer_elapsed(const boost::system::error_code &ec);
    void submithrtimer_elapsed(const boost::system::error_code &ec);
    void reconnecttimer_elapsed(const boost::system::error_code &ec);

    std::atomic<bool> m_running       = {false};
    std::atomic<bool> m_stopping      = {false};
    std::atomic<bool> m_async_pending = {false};

    unsigned m_connectionAttempt = 0;

    std::string           m_selectedHost       = ""; // Holds host name (and endpoint) of selected connection
    std::atomic<unsigned> m_connectionSwitches = {0};

    unsigned m_activeConnectionIdx = 0;

    WorkPackage m_currentWp;

    boost::asio::io_service::strand m_io_strand;
    boost::asio::deadline_timer     m_failovertimer;
    boost::asio::deadline_timer     m_submithrtimer;
    boost::asio::deadline_timer     m_reconnecttimer;

    std::unique_ptr<PoolClient> p_client = nullptr;

    std::atomic<unsigned> m_epochChanges = {0};

    static PoolManager *m_this;

    uint64_t m_u64TotalNumberOfHashes = 0;
};

} // namespace eth
} // namespace dev
