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
// Created by zeyad-osama on 25/01/2021.
//

#ifndef THOTH_EXCEPTIONS_NOT_UNDEFINED_EXCEPTION_HPP
#define THOTH_EXCEPTIONS_NOT_UNDEFINED_EXCEPTION_HPP

#include <exception>

namespace thoth {
    namespace exceptions {
        struct UndefinedException : public std::exception {
            const char *what() const noexcept override {
                return "Undefined Exception: Undefined feature.";
            }
        };
    } //namespace exceptions
} //namespace thoth

#endif //THOTH_EXCEPTIONS_NOT_UNDEFINED_EXCEPTION_HPP
