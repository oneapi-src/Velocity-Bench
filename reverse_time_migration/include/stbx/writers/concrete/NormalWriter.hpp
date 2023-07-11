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
// Created by zeyad-osama on 02/09/2020.
//

#ifndef PIPELINE_WRITERS_NORMAL_WRITER_HPP
#define PIPELINE_WRITERS_NORMAL_WRITER_HPP

#include <stbx/writers/interface/Writer.hpp>

namespace stbx {
    namespace writers {

        class NormalWriter : public Writer {
        public:
            NormalWriter() = default;

            ~NormalWriter() override = default;

        private:
            void SpecifyRawMigration() override;

            void Initialize() override {};

            void PostProcess() override;
        };
    }//namespace writers
}//namespace stbx

#endif //PIPELINE_WRITERS_NORMAL_WRITER_HPP
