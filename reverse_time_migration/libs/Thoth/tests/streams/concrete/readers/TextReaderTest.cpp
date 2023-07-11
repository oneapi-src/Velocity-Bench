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
// Created by pancee on 1/10/21.
//

#include <thoth/streams/concrete/readers/TextReader.hpp>

#include <thoth/configurations/concrete/JSONConfigurationMap.hpp>
#include <thoth/configurations/interface/MapKeys.h>

#include <libraries/catch/catch.hpp>

using namespace thoth::streams;
using namespace thoth::dataunits;
using namespace thoth::configuration;
using json = nlohmann::json;


TEST_CASE("TextReader") {
//    /*
//     * Utility settings.
//     */
//
//    std::unordered_map<TraceHeaderKey, std::string> gather_keys;
//    TraceHeaderKey sx(TraceHeaderKey::SX);
//    TraceHeaderKey gx(TraceHeaderKey::GX);
//    gather_keys[sx] = "18";
//    gather_keys[gx] = "18";
//
//    json configuration_map;
//    configuration_map[IO_K_PROPERTIES][IO_K_READ_PATH] = TEST_DATA_PATH "/synthetic/meta-data.json";
//    configuration_map[IO_K_PROPERTIES][IO_K_READ_HEADERS_ONLY] = false;
//
//    TextReader r(new JSONConfigurationMap(configuration_map));
//    r.AcquireConfiguration();
//    r.Initialize(gather_keys, nullptr);
//
//
//    Gather *test_gather = tr.GetGather();
//    std::cout << std::endl;
//    std::cout << test_gather->GetTrace(0)->GetTraceHeaderKeyValue<float>(sx) << std::endl;
}