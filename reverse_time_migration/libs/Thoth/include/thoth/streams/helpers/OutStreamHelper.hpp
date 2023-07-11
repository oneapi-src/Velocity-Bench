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

#ifndef THOTH_STREAMS_HELPERS_OUT_FILE_HELPER_HPP
#define THOTH_STREAMS_HELPERS_OUT_FILE_HELPER_HPP

#include <fstream>

namespace thoth {
    namespace streams {
        namespace helpers {

            /**
             * @brief File helper to take any stream and helps manipulate or write any regarded
             * data to it.
             */
            class OutStreamHelper {
            public:
                /**
                 * @brief Explicit constructor.
                 * @param[in] aFilePath
                 */
                explicit OutStreamHelper(std::string &aFilePath);

                /**
                 * @brief Destructor.
                 */
                ~OutStreamHelper();

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
                inline size_t
                GetFileSize() const { return this->mFileSize; };

                /**
                 * @brief Writes a block of bytes in current stream.
                 *
                 * @param[in] aData
                 * @param[in] aStartPosition
                 * @param[in] aBlockSize
                 * @return Status flag.
                 */
                int
                WriteBytesBlock(unsigned char *aData, size_t aBlockSize);

            private:
                /// File path.
                std::string mFilePath;
                /// File output stream.
                std::ofstream mOutStream;
                /// File size.
                size_t mFileSize;
            };

        } //namespace helpers
    } //namespace streams
} //namespace thoth

#endif //THOTH_STREAMS_HELPERS_OUT_FILE_HELPER_HPP
