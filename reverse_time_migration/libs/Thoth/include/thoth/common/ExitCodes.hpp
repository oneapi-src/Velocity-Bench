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
// Created by zeyad-osama on 07/03/2021.
//

#ifndef THOTH_COMMON_EXIT_CODES_HPP
#define THOTH_COMMON_EXIT_CODES_HPP

namespace thoth {
    namespace common {
        namespace exitcodes {

#define IO_RC_FAILURE        0
#define IO_RC_SUCCESS        1
#define IO_RC_ABORTED        2

        } // namespace exitcodes
    } // namespace common
} // namespace thoth

#endif //THOTH_COMMON_EXIT_CODES_HPP
