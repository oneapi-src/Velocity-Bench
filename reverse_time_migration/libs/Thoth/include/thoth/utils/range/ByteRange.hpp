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
// Created by zeyad-osama on 21/02/2021.
//

#ifndef THOTH_RANGE_BYTE_RANGE_HPP
#define THOTH_RANGE_BYTE_RANGE_HPP

namespace thoth {
    namespace utils {
        namespace range {

            struct ByteRange {
            public:
                enum OFFSET_TYPE : char {
                    SRT = 'H',
                    USRT = 'U',
                    INT = 'I',
                    UINT = 'P',
                    LNG = 'L',
                    ULNG = 'V',
                    FLT = 'F',
                    DBL = 'D',
                    CHAR = 'S'
                };

            public:
                ByteRange &operator=(ByteRange const &RHS) = delete;

                ByteRange() = default;

                ByteRange(char aOffset, int aMinByte) {
                    mOffset = aOffset;
                    mMinimumByte = aMinByte;
                }

                ByteRange(const ByteRange &aBR) {
                    mOffset = aBR.mOffset;
                    mMinimumByte = aBR.mMinimumByte;
                };

                inline int GetMinimumByte() const {
                    return this->mMinimumByte;
                }

                inline char GetOffset() const {
                    return this->mOffset;
                }

            private:
                char mOffset;
                int mMinimumByte;
            };

        } //namespace ranges
    } //namespace utils
} //namespace thoth
#endif //THOTH_RANGE_BYTE_RANGE_HPP
