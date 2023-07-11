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

#ifndef THOTH_STREAMS_IMAGE_WRITER_HPP
#define THOTH_STREAMS_IMAGE_WRITER_HPP

#include <thoth/streams/primitive/Writer.hpp>

namespace thoth {
    namespace streams {
        /**
         * @brief
         */
        class ImageWriter : public Writer {
        public:
            /**
             * @brief Destructor
             */
            ~ImageWriter() override = default;
        };

    } //streams
} //thoth

#endif //THOTH_STREAMS_IMAGE_WRITER_HPP
