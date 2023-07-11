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
// Created by zeyad-osama on 21/01/2021.
//

#ifndef THOTH_COMMON_ASSERTIONS_TPP
#define THOTH_COMMON_ASSERTIONS_TPP

#include <type_traits>

#define ASSERT_IS_POD(ARG)             std::is_pod<ARG>::value
#define ASSERT_IS_STR(ARG)             std::is_base_of<std::string, ARG>::value

#define ASSERT_T_TEMPLATE(ARG) \
static_assert(ASSERT_IS_POD(ARG) || ASSERT_IS_STR(ARG), "T type is not compatible")

#endif //THOTH_COMMON_ASSERTIONS_TPP
