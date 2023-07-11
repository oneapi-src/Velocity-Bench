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
// Created by amr-nasr on 12/01/2021.
//

#ifndef THOTH_STREAMS_WRITER_HPP
#define THOTH_STREAMS_WRITER_HPP

#include <thoth/streams/interface/Stream.hpp>
#include <thoth/configurations/interface/Configurable.hpp>
#include <thoth/data-units/concrete/Trace.hpp>
#include <thoth/data-units/concrete/Gather.hpp>
#include <thoth/data-units/data-types/TraceHeaderKey.hpp>

#include <unordered_map>
#include <vector>

namespace thoth {
    namespace streams {
        /**
         * @brief
         * Writer interface for seismic data.
         * This should be configurable to allow each reader to extract its unique additional properties.
         * A simple flow should be like the following:
         * Writer.AcquireConfiguration(...);
         * Writer.Initialize(...);
         * Do all other operations afterwards.
         */
        class Writer : public Stream {
        public:
            /**
             * @brief
             * Virtual destructor to allow correct destruction of concrete classes.
             */
            virtual ~Writer() = default;

            /**
             * @brief
             * Initializes the writer with the appropriate settings applied
             * to the writer, and any preparations needed are done.
             * Should be called once at the start.
             *
             * @param[in] aFilePath
             * The path to be used, either directly or as a seed, for writing. 
             */
            virtual int
            Initialize(std::string &aFilePath) = 0;

            /**
             * @brief
             * Does any final updates needed for consistency of the writer.
             * Release all resources and close everything.
             * Should be initialized again afterwards to be able to reuse it again.
             */
            virtual int
            Finalize() = 0;

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
            virtual int
            Write(std::vector<dataunits::Gather *> aGathers) = 0;

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
            virtual int
            Write(thoth::dataunits::Gather *aGather) = 0;
        };

    } //namespace streams
} //namespace thoth

#endif //THOTH_STREAMS_WRITER_HPP
