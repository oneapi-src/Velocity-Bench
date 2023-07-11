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
// Created by zeyad-osama on 09/09/2020.
//

#ifndef PIPELINE_WRITERS_ADCIG_WRITER_HPP
#define PIPELINE_WRITERS_ADCIG_WRITER_HPP

#include <stbx/writers/interface/Writer.hpp>

namespace stbx {
    namespace writers {

        class ADCIGWriter : public Writer {
        public:
            ADCIGWriter();

            ~ADCIGWriter() override;

            void Write(const std::string &write_path, bool is_traces) override;

        private:
            void Initialize() override;

            void PostProcess() override {};

            void Filter() override;

            void SpecifyRawMigration() override {};

            void PrepareResults();

            void WriteSegyIntervals(float *frame, const std::string &file_name);

            void WriteCIG(float *frame, const std::string &file_name);

        private:
            float *mRawMigrationIntervals;

            float *mFilteredMigrationIntervals;

            float *mRawMigrationStacked;

            float *mFilteredMigrationStacked;

            int mIntervalLength = 25;
        };
    }//namespace writers
}//namespace stbx

#endif //PIPELINE_WRITERS_ADCIG_WRITER_HPP
