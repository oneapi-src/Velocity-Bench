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
// Created by zeyad-osama on 24/01/2021.
//

#ifndef SEISMIC_TOOLBOX_TESTS_TEST_UTILS_UTILS_H
#define SEISMIC_TOOLBOX_TESTS_TEST_UTILS_UTILS_H

#include <string>

namespace stbx {
    namespace testutils {
        template<typename Base, typename T>
        inline bool instanceof(const T *) {
            return std::is_base_of<Base, T>::value;
        }
    } //namespace testutils
} //namespace stbx

#endif //SEISMIC_TOOLBOX_TESTS_TEST_UTILS_UTILS_H
