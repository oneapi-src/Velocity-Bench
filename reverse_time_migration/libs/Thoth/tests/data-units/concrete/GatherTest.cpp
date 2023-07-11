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
// Created by pancee on 11/20/20.
//

#include <thoth/data-units/concrete/Gather.hpp>

#include <thoth/data-units/concrete/Trace.hpp>
#include <thoth/data-units/data-types/TraceHeaderKey.hpp>

#include <libraries/catch/catch.hpp>

#include <vector>
#include <string>

using namespace thoth::dataunits;


TEST_CASE("GatherTest", "[Gather]") {
//    int n_s = 10;
//    int trace_num = 5;
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
//    TraceHeaderKey trace_header(TraceHeaderKey::NS);
//    std::string unique_key_value("200");
//
//    std::unordered_map<TraceHeaderKey, std::string> gather_keys;
//    gather_keys[trace_header] = unique_key_value;
//    Gather gather(trace_header, unique_key_value, gather_traces);
//
//    SECTION("Test SetUniqueKeyValue() and GetUniqueKeyValue()") {
//        TraceHeaderKey duse(TraceHeaderKey::DUSE);
//        int true_duse = 101;
//        gather.SetUniqueKeyValue<int>(duse, true_duse);
//        int duse_val = gather.GetUniqueKeyValue<int>(duse);
//        REQUIRE(duse_val == true_duse);
//    }
//
//    SECTION("Test GetTrace()") {
//        REQUIRE(gather.GetTrace(0)->GetTraceHeaderKeyValue<int>(trace_header) == n_s);
//        int i = 0;
//        for (auto &e : true_trace_data) {
//            REQUIRE(gather.GetTrace(0)->GetTraceData()[i] == e);
//            i += 1;
//        }
//    }
//
//    SECTION("Test GetNumberTraces()") {
//        REQUIRE(trace_num == gather.GetNumberTraces());
//    }
//
//    SECTION("Test RemoveTrace()") {
//        gather.RemoveTrace(trace_num);
//        REQUIRE(trace_num - 1 == gather.GetNumberTraces());
//    }
//
//    SECTION("Test AddTrace()") {
//        Trace new_trace(n_s);
//        gather.AddTrace(&new_trace);
//        trace_num += 1;
//        REQUIRE(trace_num == gather.GetNumberTraces());
//    }
//
//    SECTION("Test SortGather()") {
//        TraceHeaderKey nhs(TraceHeaderKey::NHS);
//        TraceHeaderKey gdel(TraceHeaderKey::GDEL);
//        TraceHeaderKey offset(TraceHeaderKey::OFFSET);
//
//        Trace *p_trace_1 = gather.GetTrace(0);
//        Trace *p_trace_2 = gather.GetTrace(1);
//        Trace *p_trace_3 = gather.GetTrace(2);
//        Trace *p_trace_4 = gather.GetTrace(3);
//        Trace *p_trace_5 = gather.GetTrace(4);
//
//        p_trace_1->SetTraceHeaderKeyValue<int>(gdel, 200);
//        p_trace_1->SetTraceHeaderKeyValue<short>(offset, 17);
//        p_trace_1->SetTraceHeaderKeyValue<long>(nhs, 444);
//
//        p_trace_2->SetTraceHeaderKeyValue<int>(gdel, 100);
//        p_trace_2->SetTraceHeaderKeyValue<short>(offset, 15);
//        p_trace_2->SetTraceHeaderKeyValue<long>(nhs, 333);
//
//        p_trace_3->SetTraceHeaderKeyValue<int>(gdel, 50);
//        p_trace_3->SetTraceHeaderKeyValue<short>(offset, 20);
//        p_trace_3->SetTraceHeaderKeyValue<long>(nhs, 333);
//
//        p_trace_4->SetTraceHeaderKeyValue<int>(gdel, 80);
//        p_trace_4->SetTraceHeaderKeyValue<short>(offset, 18);
//        p_trace_4->SetTraceHeaderKeyValue<long>(nhs, 222);
//
//        p_trace_5->SetTraceHeaderKeyValue<int>(gdel, 30);
//        p_trace_5->SetTraceHeaderKeyValue<short>(offset, 16);
//        p_trace_5->SetTraceHeaderKeyValue<long>(nhs, 100);
//
//        std::vector<std::pair<TraceHeaderKey, Gather::SortDirection>> sorting_keys;
//        sorting_keys.push_back(std::make_pair(nhs, Gather::SortDirection::ASC));
//        sorting_keys.push_back(std::make_pair(gdel, Gather::SortDirection::DES));
//        sorting_keys.push_back(std::make_pair(offset, Gather::SortDirection::ASC));
//        gather.SortGather(sorting_keys);
//
//        REQUIRE(gather.GetTrace(0)->GetTraceData() == p_trace_5->GetTraceData());
//        REQUIRE(gather.GetTrace(1)->GetTraceData() == p_trace_4->GetTraceData());
//        REQUIRE(gather.GetTrace(2)->GetTraceData() == p_trace_2->GetTraceData());
//        REQUIRE(gather.GetTrace(3)->GetTraceData() == p_trace_3->GetTraceData());
//        REQUIRE(gather.GetTrace(4)->GetTraceData() == p_trace_1->GetTraceData());
//    }
}
