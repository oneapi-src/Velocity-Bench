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
// Created by zeyad-osama on 09/03/2021.
//

#ifndef THOTH_HELPERS_DISPLAYERS_H
#define THOTH_HELPERS_DISPLAYERS_H

namespace thoth {
    namespace helpers {
        namespace displayers {

            /**
             * @brief Prints the text header extracted from the given SEG-Y file in the
             * SEG-Y community format.
             *
             * @param apTextHeader
             */
            void
            print_text_header(unsigned char *apTextHeader);

        } //namespace displayers
    } //namespace helpers
} //namespace thoth

#endif //THOTH_HELPERS_DISPLAYERS_H
