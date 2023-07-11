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
// Created by zeyad-osama on 01/02/2021.
//

#ifndef OPERATIONS_LIB_TEST_UTILS_NUMBERS_HELPERS_HPP
#define OPERATIONS_LIB_TEST_UTILS_NUMBERS_HELPERS_HPP

#include <cmath>

namespace operations {
    namespace testutils {

#define OP_TU_TOLERANCE 0

        float calculate_norm(const float *mat, uint nx, uint nz, uint ny);

        bool approximately_equal(float a, float b, float tolerance = OP_TU_TOLERANCE);

        bool essentially_equal(float a, float b, float tolerance = OP_TU_TOLERANCE);

        bool definitely_greater_than(float a, float b, float tolerance = OP_TU_TOLERANCE);

        bool definitely_less_than(float a, float b, float tolerance = OP_TU_TOLERANCE);

    } //namespace testutils
} //namespace operations


#endif //OPERATIONS_LIB_TEST_UTILS_NUMBERS_HELPERS_HPP
