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
// Created by zeyad-osama on 08/03/2021.
//

#ifndef THOTH_UTILS_CHECKERS_CHECKER_HPP
#define THOTH_UTILS_CHECKERS_CHECKER_HPP

namespace thoth {
    namespace utils {
        namespace checkers {

            class Checker {
            public:
                /**
                 * @brief Return 0 for big endian, 1 for little endian.
                 */
                static bool
                IsLittleEndianMachine();

            private:
                Checker() = default;

            };
        } //namespace checkers
    } //namespace utils
} //namespace thoth

#endif //THOTH_UTILS_CHECKERS_CHECKER_HPP
