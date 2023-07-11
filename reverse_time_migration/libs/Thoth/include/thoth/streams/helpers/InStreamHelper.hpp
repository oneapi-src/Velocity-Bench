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
// Created by zeyad-osama on 10/03/2021.
//

#ifndef THOTH_STREAMS_HELPERS_IN_FILE_HELPER_HPP
#define THOTH_STREAMS_HELPERS_IN_FILE_HELPER_HPP

#include <thoth/data-units/concrete/Trace.hpp>
#include <thoth/data-units/concrete/Gather.hpp>
#include <thoth/lookups/tables/TextHeaderLookup.hpp>
#include <thoth/lookups/tables/TraceHeaderLookup.hpp>
#include <thoth/lookups/tables/BinaryHeaderLookup.hpp>

#include <fstream>
#include <cstring>

namespace thoth {
    namespace streams {
        namespace helpers {

            /**
             * @brief File helper to take any stream and helps manipulate or get any regarded
             * meta data from it.
             */
            class InStreamHelper {
            public:
                /**
                 * @brief Explicit constructor.
                 * @param[in] aFilePath
                 */
                explicit InStreamHelper(std::string &aFilePath);

                /**
                 * @brief Destructor.
                 */
                ~InStreamHelper();

                /**
                 * @brief Opens stream for the file regarded.
                 * @return File size.
                 */
                size_t
                Open();

                /**
                 * @brief Release all resources and close everything.
                 */
                int
                Close();

                /**
                 * @brief Gets file size for a given file path.
                 * @return size_t
                 */
                size_t
                GetFileSize();

                /**
                 * @brief Get current position for a given file path.
                 * @return size_t
                 */
                size_t
                GetCurrentPosition();

                /**
                 * @brief Reads a block of bytes from current stream.
                 *
                 * @param[in] aStartPosition
                 * @param[in] aBlockSize
                 * @return unsigned char *
                 */
                unsigned char *
                ReadBytesBlock(size_t aStartPosition, size_t aBlockSize);

                /**
                 * @brief Reads a text header, be it the original text header or the extended text header
                 * from a given SEG-Y file, by passing the start byte position of it.
                 *
                 * @param[in] aStartPosition
                 * @return unsigned char *
                 */
                unsigned char *
                ReadTextHeader(size_t aStartPosition);

                /**
                 * @brief Reads a binary header from a given SEG-Y file, by passing the start byte position of it.
                 * @param[in] aStartPosition
                 * @return BinaryHeaderLookup
                 */
                lookups::BinaryHeaderLookup
                ReadBinaryHeader(size_t aStartPosition);

                /**
                 * @brief Reads a trace header from a given SEG-Y file, by passing the start byte position of it.
                 * @param[in] aStartPosition
                 * @return TraceHeaderLookup
                 */
                lookups::TraceHeaderLookup
                ReadTraceHeader(size_t aStartPosition);

                /**
                 * @brief Reads a single trace from a given SEG-Y file, by passing the start byte position
                 * of the desired trace, the trace header for allocation purposes and the binary header for
                 * data formats purposes.
                 *
                 * @param[in] aStartPosition
                 * @param[in] aTraceHeaderLookup
                 * @param[in] aBinaryHeaderLookup
                 * @return Trace *
                 */
                dataunits::Trace *
                ReadFormattedTraceData(size_t aStartPosition,
                                       const thoth::lookups::TraceHeaderLookup &aTraceHeaderLookup,
                                       const thoth::lookups::BinaryHeaderLookup &aBinaryHeaderLookup);

            public:
                /**
                 * @brief Gets trace data size given its header and binary header to
                 * determine whether to take NS or BHS and deduce data format.
                 *
                 * @param[in] aTraceHeaderLookup
                 * @param[in] aBinaryHeaderLookup
                 * @return
                 */
                static size_t
                GetTraceDataSize(const thoth::lookups::TraceHeaderLookup &aTraceHeaderLookup,
                                 const thoth::lookups::BinaryHeaderLookup &aBinaryHeaderLookup);

                /**
                 * @brief Gets samples number in a trace given its header and binary header to
                 * determine whether to take NS or BHS.
                 *
                 * @param[in] aTraceHeaderLookup
                 * @param[in] aBinaryHeaderLookup
                 * @return
                 */
                static size_t
                GetSamplesNumber(const lookups::TraceHeaderLookup &aTraceHeaderLookup,
                                 const lookups::BinaryHeaderLookup &aBinaryHeaderLookup);

            private:
                /// File path.
                std::string mFilePath;
                /// File input stream.
                std::ifstream mInStream;
                /// File size.
                size_t mFileSize;
            };

        } //namespace helpers
    } //namespace streams
} //namespace thoth

#endif //THOTH_STREAMS_HELPERS_IN_FILE_HELPER_HPP
