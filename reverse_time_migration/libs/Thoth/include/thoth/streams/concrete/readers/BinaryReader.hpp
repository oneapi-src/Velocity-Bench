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
// Created by zeyad-osama on 02/11/2020.
//

#ifndef THOTH_STREAMS_BINARY_READER_HPP
#define THOTH_STREAMS_BINARY_READER_HPP

#include <thoth/streams/primitive/Reader.hpp>

namespace thoth {
    namespace streams {
        /**
         * @brief
         */
        class BinaryReader : public Reader {
        public:

            /**
             * @brief Destructor
             */
            ~BinaryReader() override;
        };

    } //streams
} //thoth

#endif //THOTH_STREAMS_BINARY_READER_HPP
