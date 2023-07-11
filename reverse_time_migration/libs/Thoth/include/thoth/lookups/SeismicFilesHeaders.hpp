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
// Created by pancee on 1/14/21.
//

#ifndef THOTH_LOOKUPS_SEISMIC_FILES_HEADERS_HPP
#define THOTH_LOOKUPS_SEISMIC_FILES_HEADERS_HPP

#include <thoth/data-units/data-types/TraceHeaderKey.hpp>

#include <thoth/utils/range/ByteRange.hpp>

#include <iostream>
#include <map>

namespace thoth {
    namespace lookups {

        class SeismicFilesHeaders {
        public:
            static const std::map<dataunits::TraceHeaderKey, utils::range::ByteRange> mTraceHeadersMap;
            static const std::map<dataunits::TraceHeaderKey, utils::range::ByteRange> mBinaryHeadersMap;

        public:
            static utils::range::ByteRange GetByteRangeByKey(dataunits::TraceHeaderKey aTraceHeaderKey) {
                std::map<dataunits::TraceHeaderKey, utils::range::ByteRange>::const_iterator itFoundKey(SeismicFilesHeaders::mTraceHeadersMap.find(aTraceHeaderKey));
                if (itFoundKey == SeismicFilesHeaders::mTraceHeadersMap.end()) { // Should never reach here
                    exit(EXIT_FAILURE);
                }

                return itFoundKey->second; 
            }
        };
    }
}

#endif //THOTH_LOOKUPS_SEISMIC_FILES_HEADERS_HPP
