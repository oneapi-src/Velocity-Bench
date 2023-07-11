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
// Created by pancee on 1/20/21.
//

#ifndef THOTH_STREAMS_HELPERS_HELPERS_H
#define THOTH_STREAMS_HELPERS_HELPERS_H

#include <thoth/data-units/concrete/Trace.hpp>
#include <thoth/data-units/concrete/Gather.hpp>
#include <thoth/data-units/data-types/TraceHeaderKey.hpp>
#include <thoth/lookups/SeismicFilesHeaders.hpp>

#include <fstream>
#include <algorithm>
#include <cstring>
#include <vector>

#define TRACE_HEADERS_BYTES 240

namespace thoth {
    namespace streams {
        namespace helpers {

            bool is_little_endian_machine();

            void swap_bytes(char &aByte_1, char &aByte_2);

            template<typename T>
            int reverse_bytes(T *apNum) {
                char temp[sizeof(T)];
                std::memcpy(temp, apNum, sizeof(T));
                for (int i = 0; i < sizeof(T) / 2; ++i) {
                    swap_bytes(temp[i], temp[sizeof(T) - 1 - i]);
                }
                std::memcpy(apNum, temp, sizeof(temp));
                return 0;
            }

            template<typename T>
            int get_index(std::vector<T> &aVector, T aValue) {
                auto it = find(aVector.begin(), aVector.end(), aValue);
                if (it != aVector.end()) {
                    int index = it - aVector.begin();
                    return index;
                } else {
                    return -1;
                }
            }

            int fill_trace_headers(char *aDummyTraceHeaders, dataunits::Trace *aTrace, bool aSwapBytes);

            dataunits::TraceHeaderKey ToTraceHeaderKey(std::string &aStringKey);

        } //namespace helpers
    } //namespace streams
} //namespace thoth

#endif //THOTH_STREAMS_HELPERS_HELPERS_H
