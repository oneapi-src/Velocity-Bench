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

#include <thoth/streams/concrete/readers/SegyReader.hpp>
#include <thoth/configurations/concrete/JSONConfigurationMap.hpp>
#include <thoth/helpers/displayers.h>
#include <thoth/utils/timer/ExecutionTimer.hpp>

#include <libraries/nlohmann/json.hpp>

#include <iostream>

using namespace std;
using namespace thoth::streams;
using namespace thoth::configuration;
using namespace thoth::dataunits;
using namespace thoth::helpers;
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
    r.Initialize(gather_keys, sorting_keys, paths);

    ExecutionTimer::Evaluate([&]() {
        r.ReadAll();
    }, true);

    displayers::print_text_header(r.GetTextHeader());
    if (r.HasExtendedTextHeader()) {
        displayers::print_text_header(r.GetExtendedTextHeader());
    }
}
