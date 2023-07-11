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
// Created by pancee on 1/13/21.
//

#include <thoth/streams/concrete/writers/BinaryWriter.hpp>

#include <thoth/configurations/concrete/JSONConfigurationMap.hpp>
#include <thoth/configurations/interface/MapKeys.h>
#include <thoth/streams/concrete/readers/TextReader.hpp>

#include <libraries/catch/catch.hpp>
#include <libraries/nlohmann/json.hpp>

#include <iostream>

using namespace thoth::streams;
using namespace thoth::dataunits;
using namespace thoth::configuration;
using json = nlohmann::json;


TEST_CASE("BinaryWriter Test") {
//    std::unordered_map<TraceHeaderKey, std::string> gather_keys;
//    std::string read_file_path(TEST_DATA_PATH "/synthetic/meta-data.json");
//    TextReader tr(gather_keys, read_file_path);
//    tr.Initialize();
//    Gather *gather = tr.GetGather();
//    std::cout << "Traces number : " << gather->GetNumberTraces() << std::endl;
//
//    json configuration_map;
//    configuration_map[IO_K_PROPERTIES][IO_K_WRITE_PATH] = TEST_RESULTS_PATH;
//
//    BinaryWriter w(new JSONConfigurationMap(configuration_map));
//    w.AcquireConfiguration();
//
//    std::string write_file_path = "binary_write_test";
//    w.Initialize(write_file_path);
//    auto write = w.Write(gather);
//    REQUIRE(write == 0);
//    w.Finalize();
}
