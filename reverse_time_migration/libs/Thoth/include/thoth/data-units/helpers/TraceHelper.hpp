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
// Created by zeyad-osama on 09/03/2021.
//

#ifndef THOTH_DATA_UNITS_HELPERS_TRACE_HELPER_HPP
#define THOTH_DATA_UNITS_HELPERS_TRACE_HELPER_HPP

#include <thoth/data-units/concrete/Trace.hpp>
#include <thoth/lookups/tables/TraceHeaderLookup.hpp>
#include <thoth/lookups/tables/BinaryHeaderLookup.hpp>

namespace thoth {
    namespace dataunits {
        namespace helpers {

            /**
             * @brief Trace helper to take any trace data unit and helps manipulate
             * or get any regarded meta data from it.
             */
            class TraceHelper {
            public:
                static int
                Weight(Trace *&apTrace,
                       lookups::TraceHeaderLookup const &aTraceHeaderLookup,
                       lookups::BinaryHeaderLookup const &aBinaryHeaderLookup);

                static int
                WeightData(Trace *&apTrace,
                           lookups::TraceHeaderLookup const &aTraceHeaderLookup,
                           lookups::BinaryHeaderLookup const &aBinaryHeaderLookup);

                static int
                WeightCoordinates(Trace *&apTrace,
                                  lookups::TraceHeaderLookup const &aTraceHeaderLookup,
                                  lookups::BinaryHeaderLookup const &aBinaryHeaderLookup);
            };
        } //namespace helpers
    } //namespace dataunits
} //namespace thoth

#endif //THOTH_DATA_UNITS_HELPERS_TRACE_HELPER_HPP
