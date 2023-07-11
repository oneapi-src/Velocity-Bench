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
// Created by zeyad-osama on 11/01/2021.
//


#include <operations/utils/io/location_comparator.h>

using namespace operations::utils::io;


bool operations::utils::io::compare_location_by_source(
        const GeneralTraces &a, const GeneralTraces &b) {
    bool value;

    int sx_a = a.TraceMetaData.source_location_x;
    int sy_a = a.TraceMetaData.source_location_y;

    int sx_b = b.TraceMetaData.source_location_x;
    int sy_b = b.TraceMetaData.source_location_y;

    // Arrange ascending order in y position global, and in x position local
    if (sy_a == sy_b) {
        value = sx_a < sx_b;
    } else {
        value = sy_a < sy_b;
    }

    return value;
}

bool operations::utils::io::compare_location_by_receiver(
        const GeneralTraces &a, const GeneralTraces &b) {
    bool value;

    int rx_a = a.TraceMetaData.receiver_location_x;
    int ry_a = a.TraceMetaData.receiver_location_y;

    int rx_b = b.TraceMetaData.receiver_location_x;
    int ry_b = b.TraceMetaData.receiver_location_y;

    // Arrange ascending order in x position first, then in y position
    if (ry_a == ry_b) {
        value = rx_a < rx_b;
    } else {
        value = ry_a < ry_b;
    }

    return value;
}
