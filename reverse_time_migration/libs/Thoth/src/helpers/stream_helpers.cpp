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
// Created by pancee on 1/27/21.
//

#include <thoth/helpers/stream_helpers.h>
#include <iostream>
#include <cassert>

using namespace thoth;
using namespace thoth::streams;

bool helpers::is_little_endian_machine() {
    unsigned int i = 1;
    char *c = (char *) &i;
    if (*c) {
        return true;
    } else
        return false;
}


void helpers::swap_bytes(char &aByte_1, char &aByte_2) {
    char temp;
    temp = aByte_1;
    aByte_1 = aByte_2;
    aByte_2 = temp;
}

int helpers::fill_trace_headers(char *aDummyTraceHeaders, dataunits::Trace *aTrace, bool aSwapBytes) {
    auto trace_headers = aTrace->GetTraceHeaders();
    std::vector<dataunits::TraceHeaderKey> trace_headers_keys;
    std::vector<std::string> trace_headers_values;
    for (auto &e : *trace_headers) {
        trace_headers_keys.push_back(e.first);
        trace_headers_values.push_back(e.second);
    }
    // Pointer to last filled byte in SeismicFilesHeaders byte area
    int last_filled_byte = 0;
    // Iterate over trace header keys of a given trace
    for (auto &e : trace_headers_keys) {
        //If Trace Header is available in SU Trace Header Lookup table
        if (lookups::SeismicFilesHeaders::GetByteRangeByKey(e).GetMinimumByte() &&
            lookups::SeismicFilesHeaders::GetByteRangeByKey(e).GetOffset()) {
            auto start_byte = lookups::SeismicFilesHeaders::GetByteRangeByKey(e).GetMinimumByte();
            auto offset = lookups::SeismicFilesHeaders::GetByteRangeByKey(e).GetOffset();
            unsigned short byte_offset;
            switch (offset) {
                case 'P':
                    byte_offset = 4;
                    break;
                case 'U':
                    byte_offset = 2;
                    break;
                default:
                    return 1;
            }
            if (start_byte > last_filled_byte) {
                memset(aDummyTraceHeaders + last_filled_byte, 0, start_byte - last_filled_byte);
            }
            auto index = get_index(trace_headers_keys, e);
            assert(index > 0);
            auto trace_header_value = trace_headers_values.at(index);
            if (aSwapBytes) {
                reverse_bytes(&trace_headers_values);
            }
            memcpy(aDummyTraceHeaders + start_byte, trace_header_value.c_str(), byte_offset);
            last_filled_byte = start_byte + byte_offset;
        }
    }
    return 0;
}


dataunits::TraceHeaderKey helpers::ToTraceHeaderKey(std::string &aStringKey) {
    char key_char[aStringKey.length()];
    strcpy(key_char, aStringKey.c_str());
    for (int i = 0; i < aStringKey.length(); ++i) {
        key_char[i] = toupper(key_char[i]);
    }
    std::string upper_case_key(key_char);

    static const std::unordered_map<std::string, dataunits::TraceHeaderKey::Key> STRING_TO_ENUM = {
            {"TRACL",    dataunits::TraceHeaderKey::TRACL},
            {"TRACR",    dataunits::TraceHeaderKey::TRACR},
            {"FLDR",     dataunits::TraceHeaderKey::FLDR},
            {"TRACF",    dataunits::TraceHeaderKey::TRACF},
            {"EP",       dataunits::TraceHeaderKey::EP},
            {"CDP",      dataunits::TraceHeaderKey::CDP},
            {"CDPT",     dataunits::TraceHeaderKey::CDPT},
            {"TRID",     dataunits::TraceHeaderKey::TRID},
            {"NVS",      dataunits::TraceHeaderKey::NVS},
            {"NHS",      dataunits::TraceHeaderKey::NHS},
            {"DUSE",     dataunits::TraceHeaderKey::DUSE},
            {"Offset",   dataunits::TraceHeaderKey::OFFSET},
            {"GELEV",    dataunits::TraceHeaderKey::GELEV},
            {"SELEV",    dataunits::TraceHeaderKey::SELEV},
            {"SDEPTH",   dataunits::TraceHeaderKey::SDEPTH},
            {"GDEL",     dataunits::TraceHeaderKey::GDEL},
            {"SDEL",     dataunits::TraceHeaderKey::SDEL},
            {"SWDEP",    dataunits::TraceHeaderKey::SWDEP},
            {"DWDEP",    dataunits::TraceHeaderKey::GWDEP},
            {"SCALEL",   dataunits::TraceHeaderKey::SCALEL},
            {"SCALCO",   dataunits::TraceHeaderKey::SCALCO},
            {"SX",       dataunits::TraceHeaderKey::SX},
            {"SY",       dataunits::TraceHeaderKey::SY},
            {"GX",       dataunits::TraceHeaderKey::GX},
            {"GY",       dataunits::TraceHeaderKey::GY},
            {"COINTIT",  dataunits::TraceHeaderKey::COINTIT},
            {"WEVEL",    dataunits::TraceHeaderKey::WEVEL},
            {"SWEVEL",   dataunits::TraceHeaderKey::SWEVEL},
            {"SUT",      dataunits::TraceHeaderKey::SUT},
            {"GUT",      dataunits::TraceHeaderKey::GUT},
            {"SSTAT",    dataunits::TraceHeaderKey::SSTAT},
            {"GSTAT",    dataunits::TraceHeaderKey::GSTAT},
            {"TSTAT",    dataunits::TraceHeaderKey::TSTAT},
            {"LAGA",     dataunits::TraceHeaderKey::LAGA},
            {"LAGB",     dataunits::TraceHeaderKey::LAGB},
            {"DELRT",    dataunits::TraceHeaderKey::DELRT},
            {"MUTS",     dataunits::TraceHeaderKey::MUTS},
            {"MUTE",     dataunits::TraceHeaderKey::MUTE},
            {"DT",       dataunits::TraceHeaderKey::DT},
            {"GAIN",     dataunits::TraceHeaderKey::GAIN},
            {"IGC",      dataunits::TraceHeaderKey::IGC},
            {"IGI",      dataunits::TraceHeaderKey::IGI},
            {"CORR",     dataunits::TraceHeaderKey::CORR},
            {"SFS",      dataunits::TraceHeaderKey::SFS},
            {"SFE",      dataunits::TraceHeaderKey::SFE},
            {"SLEN",     dataunits::TraceHeaderKey::SLEN},
            {"STYP",     dataunits::TraceHeaderKey::STYP},
            {"STAS",     dataunits::TraceHeaderKey::STAS},
            {"STAE",     dataunits::TraceHeaderKey::STAE},
            {"TATYP",    dataunits::TraceHeaderKey::TATYP},
            {"AFILF",    dataunits::TraceHeaderKey::AFILF},
            {"AFILS",    dataunits::TraceHeaderKey::AFILS},
            {"NOFILF",   dataunits::TraceHeaderKey::NOFILF},
            {"NOFILS",   dataunits::TraceHeaderKey::NOFILS},
            {"LCF",      dataunits::TraceHeaderKey::LCF},
            {"HCF",      dataunits::TraceHeaderKey::HCF},
            {"LCS",      dataunits::TraceHeaderKey::LCS},
            {"HCS",      dataunits::TraceHeaderKey::HCS},
            {"YEAR",     dataunits::TraceHeaderKey::YEAR},
            {"DAY",      dataunits::TraceHeaderKey::DAY},
            {"HOUR",     dataunits::TraceHeaderKey::HOUR},
            {"MINUTE",   dataunits::TraceHeaderKey::MINUTE},
            {"SEC",      dataunits::TraceHeaderKey::SEC},
            {"TIMBAS",   dataunits::TraceHeaderKey::TIMBAS},
            {"TRWF",     dataunits::TraceHeaderKey::TRWF},
            {"GRNORS",   dataunits::TraceHeaderKey::GRNORS},
            {"GRNOFR",   dataunits::TraceHeaderKey::GRNOFR},
            {"GRNLOF",   dataunits::TraceHeaderKey::GRNLOF},
            {"GAPS",     dataunits::TraceHeaderKey::GAPS},
            {"OTRAV",    dataunits::TraceHeaderKey::OTRAV},
            {"D1",       dataunits::TraceHeaderKey::D1},
            {"F1",       dataunits::TraceHeaderKey::F1},
            {"D2",       dataunits::TraceHeaderKey::D2},
            {"F2",       dataunits::TraceHeaderKey::F2},
            {"UNGPOW",   dataunits::TraceHeaderKey::UNGPOW},
            {"UNSCALE",  dataunits::TraceHeaderKey::UNSCALE},
            {"NTR",      dataunits::TraceHeaderKey::NTR},
            {"MARK",     dataunits::TraceHeaderKey::MARK},
            {"SHORTPAD", dataunits::TraceHeaderKey::SHORTPAD},
            {"NS",       dataunits::TraceHeaderKey::NS}
    };

    std::unordered_map<std::string, dataunits::TraceHeaderKey::Key>::const_iterator itFoundKey(STRING_TO_ENUM.find(upper_case_key));
    if (itFoundKey == STRING_TO_ENUM.end()) {
        std::cerr << "Unable to find key: " << upper_case_key << std::endl;
        exit(EXIT_FAILURE);
    }

    return dataunits::TraceHeaderKey(itFoundKey->second);
}
