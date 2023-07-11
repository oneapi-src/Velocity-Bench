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
// Created by pancee on 11/10/20.
//

#include <thoth/data-units/concrete/Trace.hpp>

#include <thoth/data-units/data-types/TraceHeaderKey.hpp>

#include <libraries/catch/catch.hpp>

#include <string>

using namespace thoth::dataunits;


TEST_CASE("TraceTest", "[Trace]") {
//    const unsigned short n_s = 10;
//    Trace trace(n_s);
//    float true_trace_data[n_s] = {1.0, 2.9, 3.8, 4.9, 5.5, 6.2, 7.4, 8.9, 9.0, 10.4};
//
//    TraceHeaderKey trace_header(TraceHeaderKey::FLDR);
//
//    SECTION("Test SetTraceData() and GetTraceData()") {
//
//        float *trace_data;
//        trace_data = trace.GetTraceData();
//        for (int i = 0; i < n_s; i++) {
//            trace_data[i] = true_trace_data[i];
//        }
//
//        for (int i = 0; i < n_s; i++) {
//            REQUIRE(true_trace_data[i] == trace_data[i]);
//        }
//    }
//
//    SECTION("Test GetNumberOfSamples()") {
//        auto number_samples = trace.GetNumberOfSamples();
//        REQUIRE(number_samples == n_s);
//    }
//
//    SECTION("Test SetTraceHeaderAttributeValue and GetTraceHeaderAttributeValue") {
//        SECTION("int values") {
//            int true_fldr_value = 20;
//            trace.SetTraceHeaderKeyValue<int>(trace_header, true_fldr_value);
//            int fldr = trace.GetTraceHeaderKeyValue<int>(trace_header);
//            REQUIRE(true_fldr_value == fldr);
//        }
//
//        SECTION("Test float values") {
//            float true_fldr_value = 20.0f;
//            trace.SetTraceHeaderKeyValue<float>(trace_header, true_fldr_value);
//            auto fldr = trace.GetTraceHeaderKeyValue<float>(trace_header);
//            REQUIRE(true_fldr_value == fldr);
//        }
//
//        SECTION("Test long values") {
//            long true_fldr_value = 20;
//            trace.SetTraceHeaderKeyValue<long>(trace_header, true_fldr_value);
//            auto fldr = trace.GetTraceHeaderKeyValue<long>(trace_header);
//            REQUIRE(true_fldr_value == fldr);
//        }
//
//        SECTION("Test uint values") {
//            uint true_fldr_value = 20;
//            trace.SetTraceHeaderKeyValue<uint>(trace_header, true_fldr_value);
//            auto fldr = trace.GetTraceHeaderKeyValue<uint>(trace_header);
//            REQUIRE(true_fldr_value == fldr);
//        }
//    }
}
