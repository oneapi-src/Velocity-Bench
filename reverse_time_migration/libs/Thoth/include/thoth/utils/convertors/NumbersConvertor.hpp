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

#ifndef THOTH_UTILS_CONVERTORS_NUMBERS_CONVERTOR_HPP
#define THOTH_UTILS_CONVERTORS_NUMBERS_CONVERTOR_HPP

#include <cstring>

namespace thoth {
    namespace utils {
        namespace convertors {

            /**
             * @brief Numbers convertor that work as a convertor from any representation type
             * to another, be it byte format, or endianness for example.
             */
            class NumbersConvertor {
            public:
                /**
                 * @brief Takes pointer to char array (i.e. byte array) and converts each
                 * element of it to little endian according to the provided format.
                 *
                 * @param apSrc
                 * @param aSize
                 * @param aFormat
                 * @return Formatted array pointer.
                 */
                static char *
                ToLittleEndian(char *apSrc, size_t aSize, short aFormat);

                /**
                 * @brief Takes pointer to short array and converts each
                 * element of it to little endian.
                 *
                 * @param[in] apSrc
                 * @param[in] aSize
                 * @return Converted short array pointer
                 */
                static short *
                ToLittleEndian(short *apSrc, size_t aSize);

                /**
                 * @brief Takes short element and converts it to little endian.
                 *
                 * @param[in] aSrc
                 * @return Converted short element.
                 */
                static short
                ToLittleEndian(short aSrc);

                /**
                 * @brief Takes pointer to unsigned short array and converts each
                 * element of it to little endian.
                 *
                 * @param[in] apSrc
                 * @param[in] aSize
                 * @return Converted unsigned short array pointer
                 */
                static unsigned short *
                ToLittleEndian(unsigned short *apSrc, size_t aSize);

                /**
                 * @brief Takes unsigned short element and converts it to little endian.
                 *
                 * @param[in] aSrc
                 * @return Converted unsigned short element.
                 */
                static unsigned short
                ToLittleEndian(unsigned short aSrc);

                /**
                 * @brief Takes pointer to int array and converts each
                 * element of it to little endian.
                 *
                 * @param[in] apSrc
                 * @param[in] aSize
                 * @return Converted int array pointer
                 */
                static int *
                ToLittleEndian(int *apSrc, size_t aSize);

                /**
                 * @brief Takes int element and converts it to little endian.
                 *
                 * @param[in] aSrc
                 * @return Converted int element.
                 */
                static int
                ToLittleEndian(int aSrc);

                /**
                 * @brief Takes pointer to int array and converts each
                 * element of it to little endian.
                 *
                 * @param[in] apSrc
                 * @param[in] aSize
                 * @return Converted int array pointer
                 */
                static unsigned int *
                ToLittleEndian(unsigned int *apSrc, size_t aSize);

                /**
                 * @brief Takes unsigned int element and converts it to little endian.
                 *
                 * @param[in] aSrc
                 * @return Converted unsigned int element.
                 */
                static unsigned int
                ToLittleEndian(unsigned int aSrc);

                /**
                 * @brief Takes pointer to signed char array and converts each
                 * element of it to little endian.
                 *
                 * @param[in] apSrc
                 * @param[in] aSize
                 * @return Converted signed char array pointer
                 */
                static signed char *
                ToLittleEndian(signed char *apSrc, size_t aSize);

                /**
                 * @brief Takes signed char element and converts it to little endian.
                 *
                 * @param[in] aSrc
                 * @return Converted signed char element.
                 */
                static signed char
                ToLittleEndian(signed char aSrc);

            private:
                /**
                 * @brief Private constructor for preventing objects creation.
                 */
                NumbersConvertor() = default;
            };
        } //namespace convertors
    } //namespace utils
} //namespace thoth

#endif //THOTH_UTILS_CONVERTORS_NUMBERS_CONVERTOR_HPP
