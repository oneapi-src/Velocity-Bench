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
// Created by zeyad-osama on 12/03/2021.
//

#ifndef THOTH_LOOKUPS_MAPPERS_TRACE_HEADER_MAPPER_HPP
#define THOTH_LOOKUPS_MAPPERS_TRACE_HEADER_MAPPER_HPP

#include <thoth/lookups/tables/TraceHeaderLookup.hpp>
#include <thoth/data-units/data-types/TraceHeaderKey.hpp>

#include <cstdio>

namespace thoth {
    namespace lookups {

        class TraceHeaderMapper {
        public:
            static size_t
            GetTraceHeaderValue(const dataunits::TraceHeaderKey::Key &aTraceHeaderKey,
                                const TraceHeaderLookup &aTraceHeaderLookup);
        };

    } //namespace lookups
} //namespace thoth

#endif //THOTH_LOOKUPS_MAPPERS_TRACE_HEADER_MAPPER_HPP
