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

#include <regex>

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#include <json/json.h>

#include <libethcore/Farm.h>
#include <libethcore/Miner.h>
#include <libpoolprotocols/PoolManager.h>

using namespace dev;
using namespace dev::eth;
using namespace std::chrono;

using boost::asio::ip::tcp;

class ApiConnection
{
  public:
    ApiConnection(boost::asio::io_service::strand &_strand, int id, bool readonly, string password);

    ~ApiConnection() = default;

    void start();

    Json::Value getMinerStat1();

    using Disconnected = std::function<void(int const &)>;
    void onDisconnected(Disconnected const &_handler) { m_onDisconnected = _handler; }

    int getId() { return m_sessionId; }

    tcp::socket &socket() { return m_socket; }

  private:
    void disconnect();
    void processRequest(Json::Value &jRequest, Json::Value &jResponse);
    void recvSocketData();
    void onRecvSocketDataCompleted(
        const boost::system::error_code &ec,
        std::size_t                      bytes_transferred);
    void sendSocketData(Json::Value const &jReq, bool _disconnect = false);
    void sendSocketData(std::string const &_s, bool _disconnect = false);
    void onSendSocketDataCompleted(const boost::system::error_code &ec, bool _disconnect = false);

    Json::Value getMinerStatDetail();
    Json::Value getMinerStatDetailPerMiner(const TelemetryType &_t, std::shared_ptr<Miner> _miner);

    std::string getHttpMinerStatDetail();

    Disconnected m_onDisconnected;

    int m_sessionId;

    tcp::socket                      m_socket;
    boost::asio::io_service::strand &m_io_strand;
    boost::asio::streambuf           m_sendBuffer;
    boost::asio::streambuf           m_recvBuffer;
    Json::StreamWriterBuilder        m_jSwBuilder;

    std::string m_message; // The internal message string buffer

    bool        m_readonly = false;
    std::string m_password = "";

    bool m_is_authenticated = true;
};

class ApiServer
{
  public:
    ApiServer(std::string const &address, int portnum, std::string const &password);
    bool isRunning() { return m_running.load(std::memory_order_relaxed); };
    void start();
    void stop();

  private:
    void begin_accept();
    void handle_accept(std::shared_ptr<ApiConnection> session, boost::system::error_code ec);

    int lastSessionId = 0;

    std::thread                                 m_workThread;
    std::atomic<bool>                           m_readonly = {false};
    std::string                                 m_password = "";
    std::atomic<bool>                           m_running  = {false};
    string                                      m_address;
    uint16_t                                    m_portnumber;
    tcp::acceptor                               m_acceptor;
    boost::asio::io_service::strand             m_io_strand;
    std::vector<std::shared_ptr<ApiConnection>> m_sessions;
};
