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
// Created by zeyad-osama on 08/03/2021.
//

#ifndef THOTH_UTILS_CONVERTORS_DATA_FORMATTER_HPP
#define THOTH_UTILS_CONVERTORS_DATA_FORMATTER_HPP

#include <cstring>

namespace thoth {
    namespace utils {
        namespace convertors {

            /**
             * @brief Floating point formatter that work as a convertor from any floating type
             * representation to another.
             */
            class FloatingPointFormatter {
            public:
                /**
                 * @brief Trace size in bytes given the NS from trace header and format code from
                 * binary header.
                 *
                 * @param[int] aSamplesNumber
                 * @param[int] aFormatCode
                 * @return Trace Size
                 */
                static int
                GetFloatArrayRealSize(unsigned short int aSamplesNumber, unsigned short int aFormatCode);

                static int
                Format(char *apSrc, size_t aSrcSize, size_t aSamplesNumber, short aFormat);

            private:
                FloatingPointFormatter() = default;

            private:
                /**
                 * @brief Converts between 32 bit IBM floating numbers to native floating number.
                 * @param[in,out] apSrc
                 * @param[in] aSrcSize
                 * @return Flag. 1 if success and 0 if conversion failed.
                 */
                static int
                FromIBM(unsigned char *&apSrc, size_t aSrcSize, size_t aSamplesNumber);

                /**
                 * @brief Converts between 64 bit IBM floating numbers to native floating number.
                 * @param[in,out] apSrc
                 * @param[in] aSize
                 * @return Flag. 1 if success and 0 if conversion failed.
                 */
                static int
                FromLong(unsigned char *&apSrc, size_t aSize, size_t aSamplesNumber);

                /**
                 * @brief Converts between 8 bit IBM floating numbers to native floating number.
                 * @param[in,out] apSrc
                 * @param[in] aSize
                 * @return Flag. 1 if success and 0 if conversion failed.
                 */
                static int
                FromShort(unsigned char *&apSrc, size_t aSize, size_t aSamplesNumber);

                /**
                 * @brief Converts between 8 bit IEEE and IEEE floating numbers.
                 * @param[in,out] apSrc
                 * @param[in] aSize
                 * @return Flag. 1 if success and 0 if conversion failed.
                 */
                static int
                FromIEEE(unsigned char *&apSrc, size_t aSize, size_t aSamplesNumber);

                /**
                 * @brief Converts between 8 bit character and IEEE floating numbers.
                 * @param[in,out] apSrc
                 * @param[in] aSize
                 * @return Flag. 1 if success and 0 if conversion failed.
                 */
                static int
                FromChar(unsigned char *&apSrc, size_t aSize, size_t aSamplesNumber);
            };
        } //namespace convertors
    } //namespace utils
} //namespace thoth

#endif //THOTH_UTILS_CONVERTORS_DATA_FORMATTER_HPP
