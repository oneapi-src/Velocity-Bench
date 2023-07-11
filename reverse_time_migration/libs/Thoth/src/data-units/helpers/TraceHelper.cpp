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

#include <thoth/data-units/helpers/TraceHelper.hpp>

#include <thoth/common/ExitCodes.hpp>
#include <thoth/utils/convertors/NumbersConvertor.hpp>

#include <cmath>

using namespace thoth::dataunits;
using namespace thoth::dataunits::helpers;
using namespace thoth::lookups;
using namespace thoth::common::exitcodes;
using namespace thoth::utils::convertors;


int TraceHelper::Weight(Trace *&apTrace,
                        TraceHeaderLookup const &aTraceHeaderLookup,
                        BinaryHeaderLookup const &aBinaryHeaderLookup) {
    return TraceHelper::WeightData(apTrace, aTraceHeaderLookup, aBinaryHeaderLookup) &&
           TraceHelper::WeightCoordinates(apTrace, aTraceHeaderLookup, aBinaryHeaderLookup);
}

int TraceHelper::WeightData(Trace *&apTrace,
                            TraceHeaderLookup const &aTraceHeaderLookup,
                            BinaryHeaderLookup const &aBinaryHeaderLookup) {
    auto format = NumbersConvertor::ToLittleEndian(aBinaryHeaderLookup.FORMAT);
    /* Scale data. */
    if (!(format == 1 || format == 5)) {
        auto trwf = NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.TRWF);
        if (trwf != 0) {
            float scale = std::pow(2.0, -trwf);
            auto data = apTrace->GetTraceData();
            for (int i = 0; i < apTrace->GetNumberOfSamples(); ++i) {
                data[i] *= scale;
            }
            apTrace->SetTraceData(data);
        }
    }
    return IO_RC_SUCCESS;
}

int TraceHelper::WeightCoordinates(Trace *&apTrace,
                                   TraceHeaderLookup const &aTraceHeaderLookup,
                                   BinaryHeaderLookup const &aBinaryHeaderLookup) {
    /* Scale coordinate. */
    auto scale_coordinate = apTrace->GetTraceHeaderKeyValue<int16_t>(TraceHeaderKey::SCALCO);
    if (scale_coordinate == 0) {
        scale_coordinate = 1;
    }
    if (scale_coordinate > 0) {
        apTrace->SetTraceHeaderKeyValue(TraceHeaderKey::SX,
                                        apTrace->GetTraceHeaderKeyValue<float>(TraceHeaderKey::SX) *
                                        scale_coordinate);
        apTrace->SetTraceHeaderKeyValue(TraceHeaderKey::SY,
                                        apTrace->GetTraceHeaderKeyValue<float>(TraceHeaderKey::SY) *
                                        scale_coordinate);
        apTrace->SetTraceHeaderKeyValue(TraceHeaderKey::GX,
                                        apTrace->GetTraceHeaderKeyValue<float>(TraceHeaderKey::GX) *
                                        scale_coordinate);
        apTrace->SetTraceHeaderKeyValue(TraceHeaderKey::GY,
                                        apTrace->GetTraceHeaderKeyValue<float>(TraceHeaderKey::GY) *
                                        scale_coordinate);
    } else {
        scale_coordinate *= -1;
        apTrace->SetTraceHeaderKeyValue(TraceHeaderKey::SX,
                                        apTrace->GetTraceHeaderKeyValue<float>(TraceHeaderKey::SX) /
                                        scale_coordinate);
        apTrace->SetTraceHeaderKeyValue(TraceHeaderKey::SY,
                                        apTrace->GetTraceHeaderKeyValue<float>(TraceHeaderKey::SY) /
                                        scale_coordinate);
        apTrace->SetTraceHeaderKeyValue(TraceHeaderKey::GX,
                                        apTrace->GetTraceHeaderKeyValue<float>(TraceHeaderKey::GX) /
                                        scale_coordinate);
        apTrace->SetTraceHeaderKeyValue(TraceHeaderKey::GY,
                                        apTrace->GetTraceHeaderKeyValue<float>(TraceHeaderKey::GY) /
                                        scale_coordinate);
    }
    apTrace->SetTraceHeaderKeyValue(TraceHeaderKey::SCALCO, 1);
    return IO_RC_SUCCESS;
}
