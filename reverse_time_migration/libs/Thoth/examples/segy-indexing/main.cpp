/*
 * Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU Lesser General Public License v3.0 or later
 * 
 * If a copy of the license was not distributed with this file, you can obtain one at 
 * https://www.gnu.org/licenses/lgpl-3.0-standalone.html
 * 
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


//
// Created by zeyad-osama on 07/03/2021.
//

#include <thoth/api/thoth.hpp>
#include <thoth/configurations/concrete/JSONConfigurationMap.hpp>
#include <thoth/utils/timer/ExecutionTimer.hpp>

#include <libraries/nlohmann/json.hpp>

#include <iostream>

using namespace std;
using namespace thoth::streams;
using namespace thoth::configuration;
using namespace thoth::dataunits;
using namespace thoth::utils::timer;
using json = nlohmann::json;


int main(int argc, char *argv[]) {
    json configuration_map;
    configuration_map[IO_K_PROPERTIES][IO_K_TEXT_HEADERS_ONLY] = false;
    configuration_map[IO_K_PROPERTIES][IO_K_TEXT_HEADERS_STORE] = false;

    std::vector<TraceHeaderKey> gather_keys = {TraceHeaderKey::FLDR};
    std::vector<std::pair<TraceHeaderKey, Gather::SortDirection>> sorting_keys;
    std::vector<std::string> paths = {DATA_PATH "/shots0601_0800.segy",
                                      DATA_PATH "/vel_z6.25m_x12.5m_exact.segy"};

    SegyReader r(new JSONConfigurationMap(configuration_map));
    r.AcquireConfiguration();

    /* Initializing + Indexing. */
    std::cout << std::endl << "Initializing + Indexing:" << std::endl;
    ExecutionTimer::Evaluate([&]() {
        r.Initialize(gather_keys, sorting_keys, paths);
    }, true);


    /* Normal case. */
    std::cout << std::endl << "Normal reading case [1]:" << std::endl;
    ExecutionTimer::Evaluate([&]() {
        r.Read({"604"});
    }, true);

    /* Normal case. */
    std::cout << std::endl << "Normal reading case [2]:" << std::endl;
    ExecutionTimer::Evaluate([&]() {
        vector<string> vec = {"729"};
        r.Read(vec);
    }, true);

    /* Not available case. */
    std::cout << std::endl << "Not available reading case:" << std::endl;
    ExecutionTimer::Evaluate([&]() {
        vector<string> vec = {"6329"};
        r.Read(vec);
    }, true);

    /* Get gather number */
    std::cout << std::endl << "Get gather number case:" << std::endl;
    unsigned int num;
    ExecutionTimer::Evaluate([&]() {
        num = r.GetNumberOfGathers();
    }, true);
    std::cout << "Number of gathers : " << num << std::endl;

    /* Get gather number */
    std::cout << std::endl << "Get gather number case:" << std::endl;
    std::vector<std::vector<std::string>> keys;
    ExecutionTimer::Evaluate([&]() {
        keys = r.GetIdentifiers();
    }, true);
    std::cout << "Number of gathers : " << keys.size() << std::endl;

    /* Finalize and closes all opened internal streams. */
    r.Finalize();
}
