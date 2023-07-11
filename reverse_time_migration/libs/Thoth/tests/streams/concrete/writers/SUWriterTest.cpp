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
// Created by pancee on 1/18/21.
//

#include <thoth/streams/concrete/writers/SUWriter.hpp>

#include <thoth/configurations/concrete/JSONConfigurationMap.hpp>
#include <thoth/configurations/interface/MapKeys.h>

#include <libraries/catch/catch.hpp>
#include <libraries/nlohmann/json.hpp>

using namespace thoth::streams;
using namespace thoth::dataunits;
using namespace thoth::configuration;
using json = nlohmann::json;


TEST_CASE("SUWriter Test") {
//    json configuration_map;
//    configuration_map[IO_K_PROPERTIES][IO_K_WRITE_LITTLE_ENDIAN] = true;
//    configuration_map[IO_K_PROPERTIES][IO_K_WRITE_PATH] = TEST_RESULTS_PATH;
//
//    SUWriter w(new JSONConfigurationMap(configuration_map));
//    w.AcquireConfiguration();
//
//    int n_s = 10;
//
//    Trace trace_1(n_s);
//    Trace trace_2(n_s);
//    Trace trace_3(n_s);
//    Trace trace_4(n_s);
//    Trace trace_5(n_s);
//    std::vector<Trace *> gather_traces;
//    float true_trace_data[] = {1.0, 2.0, 4.1, 5.2, 6.4, 7.5, 8.9, 9.8, 10.0, 9.0};
//    auto p_trace_data = trace_1.GetTraceData();
//    for (int i = 0; i < n_s; i++) {
//        p_trace_data[i] = true_trace_data[i];
//    }
//
//    gather_traces.push_back(&trace_1);
//    gather_traces.push_back(&trace_2);
//    gather_traces.push_back(&trace_3);
//    gather_traces.push_back(&trace_4);
//    gather_traces.push_back(&trace_5);
//    TraceHeaderKey trace_header(TraceHeaderKey::FLDR);
//    std::string unique_key_value("200");
//
//    std::unordered_map<TraceHeaderKey, std::string> gather_keys;
//    gather_keys[trace_header] = unique_key_value;
//    Gather gather(trace_header, unique_key_value, gather_traces);
//
//    std::string write_file_path = "su_test";
//    w.Initialize(write_file_path);
//    auto write = w.Write(&gather);
//    REQUIRE(write == 0);
//    w.Finalize();
}