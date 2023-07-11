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
// Created by zeyad-osama on 24/09/2020.
//

#ifndef OPERATIONS_LIB_EXCEPTIONS_AXIS_EXCEPTION_H
#define OPERATIONS_LIB_EXCEPTIONS_AXIS_EXCEPTION_H

#include <exception>

namespace operations {
    namespace exceptions {
        struct AxisException : public std::exception {
            const char *what() const noexcept override {
                return "Axis Exception: Axis is not in range of [0-2] i.e. x,y,z";
            }
        };

/**
 * @brief Checks if the provided axe is within the x,y,z range.
 * @param axis[in]
 * @return boolean[out]
 */
        bool inline is_out_of_range(uint axis) {
            return axis > 2;
        }
    } //namespace exceptions
} //namespace operations

#endif //OPERATIONS_LIB_EXCEPTIONS_AXIS_EXCEPTION_H
