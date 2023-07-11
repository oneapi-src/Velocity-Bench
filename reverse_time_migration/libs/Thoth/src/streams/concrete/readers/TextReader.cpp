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
// Created by zeyad-osama on 02/11/2020.
//

#include <thoth/streams/concrete/readers/TextReader.hpp>
#include <thoth/data-units/data-types/TraceHeaderKey.hpp>
#include <thoth/common/ExitCodes.hpp>
#include <cassert>

/// @todo To be removed
/// {
#include <thoth/exceptions/Exceptions.hpp>

using namespace thoth::exceptions;
/// }

using namespace thoth::streams;
using namespace thoth::generators;
using namespace thoth::dataunits;;
using namespace thoth::common::exitcodes;


TextReader::TextReader(thoth::configuration::ConfigurationMap *apConfigurationMap) :
        mpSyntheticModelGenerator(new SyntheticModelGenerator(apConfigurationMap)) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mEnableHeaderOnly = false;
    this->mpGather = nullptr;
}

TextReader::~TextReader() {
    delete this->mpSyntheticModelGenerator;
}

void TextReader::AcquireConfiguration() {
    this->mFilePath = this->mpConfigurationMap->GetValue(
            IO_K_PROPERTIES, IO_K_READ_PATH, this->mFilePath);
    this->mEnableHeaderOnly = this->mpConfigurationMap->GetValue(
            IO_K_PROPERTIES, IO_K_TEXT_HEADERS_ONLY, this->mEnableHeaderOnly);
}

std::string TextReader::GetExtension() {
    return IO_K_EXT_IMG;
}

int TextReader::Initialize(std::vector<std::string> &aGatherKeys,
                           std::vector<std::pair<std::string, dataunits::Gather::SortDirection>> &aSortingKeys,
                           std::vector<std::string> &aPaths) {
    for (auto &e : aGatherKeys) {
        auto gather_key = helpers::ToTraceHeaderKey(e);
        this->mGatherKeys.push_back(gather_key);
    }
    for (auto &e : aSortingKeys) {
        auto key = helpers::ToTraceHeaderKey(e.first);
        auto sort_dir = e.second;
        this->mSortingKeys.push_back(std::make_pair(key, sort_dir));
    }
    this->mpSyntheticModelGenerator->Generate();
    this->mpSyntheticModelGenerator->BuildGather();
    this->mpGather = this->mpSyntheticModelGenerator->GetGather();
    return IO_RC_SUCCESS;
}

int TextReader::Initialize(std::vector<TraceHeaderKey> &aGatherKeys,
                           std::vector<std::pair<TraceHeaderKey, dataunits::Gather::SortDirection>> &aSortingKeys,
                           std::vector<std::string> &aPaths) {
    /// @todo To be removed
    /// {
    throw NotImplementedException();
    /// }
}

int TextReader::Finalize() {
    return IO_RC_SUCCESS;
}

void TextReader::SetHeaderOnlyMode(bool aEnableHeaderOnly) {
    this->mEnableHeaderOnly = aEnableHeaderOnly;
}

unsigned int TextReader::GetNumberOfGathers() {
    auto unique_identifiers = this->GetIdentifiers();
    return unique_identifiers.size();
}

std::vector<std::vector<std::string>> TextReader::GetIdentifiers() {
    auto traces_size = this->mpGather->GetNumberTraces();
    std::vector<std::vector<std::string>> unique_identifiers;
    for (int i = 0; i < traces_size; ++i) {
        auto trace = this->mpGather->GetTrace(i);
        auto trace_headers = trace->GetTraceHeaders();
        std::vector<std::string> trace_gathering_keys;
        for (auto &e : this->mGatherKeys) {
            auto gthr_k_trc_hdr_val = &trace_headers->at(e);
            trace_gathering_keys.push_back(*gthr_k_trc_hdr_val);
        }
        if (std::find(unique_identifiers.begin(), unique_identifiers.end(), trace_gathering_keys) ==
            unique_identifiers.end()) {
            // Trace Headers Not Found
            unique_identifiers.push_back(trace_gathering_keys);
        }
    }
    return unique_identifiers;
}

std::vector<Gather *> TextReader::ReadAll() {
    auto traces_size = this->mpGather->GetNumberTraces();

    // Get unique identifiers of all gather in file.
    auto unique_identifiers = this->GetIdentifiers();

    // Vector to be returned.
    std::vector<Gather *> return_gathers;

    // Each gather and its identifiers will be at the same offset in each of unique_identifiers
    // and return_gathers vectors.
    return_gathers.reserve(unique_identifiers.size());

    // For each element/vector in file unique identifiers
    for (int k = 0; k < unique_identifiers.size(); ++k) {
        std::unordered_map<TraceHeaderKey, std::string> gather_identifiers;

        // Get gather ids at offset k
        auto id = unique_identifiers.at(k);

        // Construct Gathering keys map for current gather.
        for (int i = 0; i < id.size(); ++i) {
            auto trc_hdr_k = this->mGatherKeys.at(i);
            auto trc_hdr_val = i;
            gather_identifiers[trc_hdr_k] = std::to_string(trc_hdr_val);
        }
        //Allocate gather at offset k in return_gather vector with its corresponding gathering keys
        //map.
        return_gathers.at(k) = new Gather(gather_identifiers);
    }

    int gather_index = 0;
    // For each trace in file
    for (int i = 0; i < traces_size; ++i) {
        auto trace = this->mpGather->GetTrace(i);

        // Get trace headers ((key, value) pairs) at index i.
        auto trace_headers = trace->GetTraceHeaders();

        // Get identifiers (gathering keys values) of the trace; in trace_gathering_keys.
        std::vector<std::string> trace_gathering_keys;
        for (auto &e : this->mGatherKeys) {
            auto gthr_k_trc_hdr_val = &trace_headers->at(e);
            trace_gathering_keys.push_back(*gthr_k_trc_hdr_val);
        }

        // If trace identifiers are valid/within whole file identifiers.
        if (std::find(unique_identifiers.begin(), unique_identifiers.end(), trace_gathering_keys)
            != unique_identifiers.end()) {
            // Found Trace Headers in Unique Identifiers
            gather_index = helpers::get_index(unique_identifiers, trace_gathering_keys);
            assert(gather_index > 0);
            return_gathers.at(gather_index)->AddTrace(trace);
            continue;
        } else {
            //Not Found Trace Headers
        }
    }
    return return_gathers;
}

std::vector<Gather *> TextReader::Read(std::vector<std::vector<std::string>> aHeaderValues) {
    auto traces_size = this->mpGather->GetNumberTraces();

    // Vector to be returned.
    std::vector<Gather *> return_gathers;

    // Each gather and its identifiers will be at the same offset in each of aHeaderValues
    // and return_gathers vectors.
    return_gathers.reserve(aHeaderValues.size());

    // For each element/vector in file aHeaderValues
    for (int k = 0; k < aHeaderValues.size(); ++k) {
        std::unordered_map<TraceHeaderKey, std::string> gather_identifiers;

        // Get gather ids at offset k
        auto id = aHeaderValues.at(k);

        // Construct Gathering keys map for current gather.
        for (int i = 0; i < id.size(); ++i) {
            auto trc_hdr_k = this->mGatherKeys.at(i);
            auto trc_hdr_val = i;
            gather_identifiers[trc_hdr_k] = std::to_string(trc_hdr_val);
        }
        // Allocate gather at offset k in return_gather vector with it's
        // corresponding gathering keys map.
        return_gathers.at(k) = new Gather(gather_identifiers);
    }

    int gather_index = 0;
    //For each trace in file
    for (int i = 0; i < traces_size; ++i) {
        auto trace = this->mpGather->GetTrace(i);

        // Get trace headers ((key, value) pairs) at index i.
        auto trace_headers = trace->GetTraceHeaders();
        // Get identifiers (gathering keys values) of the trace; in trace_gathering_keys.
        std::vector<std::string> trace_gathering_keys;
        for (auto &e : this->mGatherKeys) {
            auto gthr_k_trc_hdr_val = &trace_headers->at(e);
            trace_gathering_keys.push_back(*gthr_k_trc_hdr_val);
        }
        // If trace identifiers are valid/within whole file identifiers.
        if (std::find(aHeaderValues.begin(), aHeaderValues.end(), trace_gathering_keys)
            != aHeaderValues.end()) {
            // Found Trace Headers in Unique Identifiers
            gather_index = helpers::get_index(aHeaderValues, trace_gathering_keys);
            assert(gather_index > 0);
            return_gathers.at(gather_index)->AddTrace(trace);
            continue;
        } else {
            // Not Found Trace Headers
        }
    }
    return return_gathers;
}

Gather *TextReader::Read(std::vector<std::string> aHeaderValues) {
    auto traces_size = this->mpGather->GetNumberTraces();
    Gather *return_gather;
    return_gather = nullptr;
    std::vector<Trace *> traces;
    bool add_trace = true;
    for (int i = 0; i < traces_size; ++i) {
        auto trace = this->mpGather->GetTrace(i);
        auto trc_hdrs = trace->GetTraceHeaders();
        for (auto &e : this->mGatherKeys) {
            auto k_index = helpers::get_index(this->mGatherKeys, e);
            assert(k_index > 0);
            auto trc_hdr_val = &trc_hdrs->at(e);
            if (aHeaderValues.at(k_index) != *trc_hdr_val) {
                add_trace = false;
            }
        }
        if (add_trace) {
            traces.push_back(trace);
        }
    }
    if (!traces.empty()) {
        std::unordered_map<TraceHeaderKey, std::string> gather_keys;
        for (int i = 0; i < this->mGatherKeys.size(); ++i) {
            gather_keys[this->mGatherKeys.at(i)] = aHeaderValues[i];
        }
        return_gather = new Gather(gather_keys, traces);
    }
    return return_gather;
}

Gather *TextReader::Read(unsigned int aIndex) {
    Gather *return_gather = nullptr;
    auto all_gathers = this->ReadAll();
    if (aIndex <= all_gathers.size()) {
        return_gather = all_gathers.at(aIndex);
    }
    return return_gather;
}
