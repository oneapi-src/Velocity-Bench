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

#include <thoth/streams/concrete/writers/CSVWriter.hpp>

#include <thoth/configurations/concrete/JSONConfigurationMap.hpp>
#include <thoth/streams/concrete/readers/TextReader.hpp>

#include <libraries/catch/catch.hpp>
#include <libraries/nlohmann/json.hpp>

using namespace thoth::streams;
using namespace thoth::dataunits;
using namespace thoth::configuration;
using json = nlohmann::json;


TEST_CASE("CSVWriter Test") {
//    json configuration_map;
//    configuration_map[IO_K_PROPERTIES][IO_K_WRITE_PATH] = TEST_RESULTS_PATH;
//
//    CSVWriter w(new JSONConfigurationMap(configuration_map));
//    w.AcquireConfiguration();
//
//    std::unordered_map<TraceHeaderKey, std::string> gather_keys;
//    std::string read_file_path = TEST_DATA_PATH "/synthetic/meta-data.json";
//
//    json configuration_map_tr;
//    configuration_map_tr[IO_K_PROPERTIES][IO_K_READ_PATH] = TEST_DATA_PATH "/synthetic/meta-data.json";
//
//    TextReader tr(new JSONConfigurationMap(configuration_map));
//    tr.AcquireConfiguration();
//    tr.Initialize();
//    Gather *gather = tr.GetGather();
//
//    std::string write_file_path = "csv_write_test";
//    w.Initialize(write_file_path);
//    auto write = w.Write(gather);
//    REQUIRE(write == 0);
//    w.Finalize();
}