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
// Created by zeyad-osama on 13/03/2021.
//

#ifndef THOTH_UTILS_CONVERTORS_STRINGS_CONVERTOR_HPP
#define THOTH_UTILS_CONVERTORS_STRINGS_CONVERTOR_HPP

#include <string>

namespace thoth {
    namespace utils {
        namespace convertors {

            /**
             * @brief Strings convertor that work as a convertor from any representation type
             * to another, be it byte format, or endianness for example.
             */
            class StringsConvertor {
            public:
                /**
                 * @brief Takes a string and converts its content to short.
                 *
                 * @param[in] aStr
                 * @return Converted string to short.
                 */
                static
                short ToShort(const std::string& aStr);

                /**
                 * @brief Takes a string and converts its content to unsigned short.
                 *
                 * @param[in] aStr
                 * @return Converted string to unsigned short.
                 */
                static
                unsigned short ToUShort(const std::string& aStr);

                /**
                 * @brief Takes a string and converts its content to int.
                 *
                 * @param[in] aStr
                 * @return Converted string to int.
                 */
                static
                int ToInt(const std::string& aStr);

                /**
                 * @brief Takes a string and converts its content to unsigned int.
                 *
                 * @param[in] aStr
                 * @return Converted string to unsigned int.
                 */
                static
                unsigned int ToUInt(const std::string& aStr);

                /**
                 * @brief Takes a string and converts its content to long.
                 *
                 * @param[in] aStr
                 * @return Converted string to long.
                 */
                static
                long ToLong(const std::string& aStr);

                /**
                 * @brief Takes a string and converts its content to unsigned long.
                 *
                 * @param[in] aStr
                 * @return Converted string to unsigned long.
                 */
                static
                unsigned long ToULong(const std::string& aStr);

                /**
                 * @brief Takes pointer to unsigned char array and converts each
                 * element from EBCDIC to ASCII.
                 *
                 * @param[in] apSrc
                 * @param[in] aSize
                 * @return Converted unsigned char array pointer
                 */
                static unsigned char *
                E2A(unsigned char *apSrc, size_t aSize);

                /**
                 * @brief Takes char element and converts it from EBCDIC to ASCII.
                 *
                 * @param[in] aSrc
                 * @return Converted char element.
                 */
                static unsigned char
                E2A(unsigned char aSrc);

            private:
                /**
                 * @brief Private constructor for preventing objects creation.
                 */
                StringsConvertor() = default;

            private:
                static unsigned char mE2ATable[256];
            };
        } //namespace convertors
    } //namespace utils
} //namespace thoth

#endif //THOTH_UTILS_CONVERTORS_STRINGS_CONVERTOR_HPP
