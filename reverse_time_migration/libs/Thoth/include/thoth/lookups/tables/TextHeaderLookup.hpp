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

#ifndef THOTH_LOOKUPS_TEXT_HEADERS_LOOKUP_HPP
#define THOTH_LOOKUPS_TEXT_HEADERS_LOOKUP_HPP

namespace thoth {
    namespace lookups {

#define IO_SIZE_TEXT_HEADER             3200    /* Size of the text header in SEG-Y files */
#define IO_SIZE_EXT_TEXT_HEADER         3200    /* Size of the extended (i.e. If available) text header in SEG-Y files */
#define IO_POS_S_TEXT_HEADER            0       /* Start position of text header in SEG-Y and SU files */
#define IO_POS_E_TEXT_HEADER            3200    /* End position of text header in SEG-Y and SU files */
#define IO_POS_S_EXT_TEXT_HEADER        3600    /* Start position of extended text header in SEG-Y and SU files */
#define IO_POS_E_EXT_TEXT_HEADER        6800    /* End position of extended text header in SEG-Y and SU files */

    } //namespace lookups
} //namespace thoth

#endif //THOTH_LOOKUPS_TEXT_HEADERS_LOOKUP_HPP
