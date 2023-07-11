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

#ifndef OPERATIONS_LIB_UTILS_READ_LOCATION_COMPARATOR_H
#define OPERATIONS_LIB_UTILS_READ_LOCATION_COMPARATOR_H

#include <seismic-io-framework/datatypes.h>

namespace operations {
    namespace utils {
        namespace io {
            bool compare_location_by_source(const GeneralTraces &a, const GeneralTraces &b);

            bool compare_location_by_receiver(const GeneralTraces &a, const GeneralTraces &b);
        } //namespace io
    } //namespace utils
} //namespace operations

#endif //OPERATIONS_LIB_UTILS_READ_LOCATION_COMPARATOR_H
