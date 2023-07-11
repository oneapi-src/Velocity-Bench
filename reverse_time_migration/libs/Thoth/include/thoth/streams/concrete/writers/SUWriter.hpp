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

#ifndef THOTH_STREAMS_SU_WRITER_HPP
#define THOTH_STREAMS_SU_WRITER_HPP

#include <thoth/streams/primitive/Writer.hpp>
#include <thoth/lookups/SeismicFilesHeaders.hpp>

#include <fstream>

namespace thoth {
    namespace streams {
        /**
         * @brief
         */
        class SUWriter : public Writer {
        public:
            explicit SUWriter(thoth::configuration::ConfigurationMap *apConfigurationMap);

            ~SUWriter() override = default;

            /**
             * @brief
             * Acquires the component configurations from a given configurations map.
             *
             * @param[in] apConfigurationMap
             * The configurations map to be used.
             */
            void
            AcquireConfiguration() override;

            /**
             * @brief Returns the file format extension of the current stream.
             */
            std::string GetExtension() override;

            /**
             * @brief
             * Initializes the writer with the appropriate settings applied
             * to the writer, and any preparations needed are done.
             * Should be called once at the start.
             *
             * @param[in] aFilePath
             * The path to be used, either directly or as a seed, for writing.
             */
            int
            Initialize(std::string &aFilePath) override;

            /**
             * @brief
             * Does any final updates needed for consistency of the writer.
             * Release all resources and close everything.
             * Should be initialized again afterwards to be able to reuse it again.
             */
            int
            Finalize() override;

            /**
             * @brief
             * Writes a group of gathers to the output stream of the writer.
             *
             * @param[in] aGathers
             * List of gathers to be written.
             *
             * @return
             * An error flag, if 0 that means operation was successful, otherwise indicate an error.
             */
            int
            Write(std::vector<dataunits::Gather *> aGathers) override;

            /**
             * @brief
             * Writes a gather to the output stream of the writer.
             *
             * @param[in] aGather
             * The gather to be written.
             *
             * @return
             * An error flag, if 0 that means operation was successful, otherwise indicate an error.
             */
            int
            Write(thoth::dataunits::Gather *aGather) override;

        private:
            std::string mFilePath;
            std::ofstream mOutputStream;
            bool mWriteLittleEndian;
        };

    } //streams
} //thoth

#endif //THOTH_STREAMS_SU_WRITER_HPP
