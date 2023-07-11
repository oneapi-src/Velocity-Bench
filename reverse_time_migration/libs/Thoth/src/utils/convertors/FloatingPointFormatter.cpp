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

#include <thoth/utils/convertors/FloatingPointFormatter.hpp>

#include <thoth/utils/checkers/Checker.hpp>
#include <thoth/exceptions/UnsupportedFeatureException.hpp>

using namespace thoth::utils::convertors;
using namespace thoth::utils::checkers;


int FloatingPointFormatter::GetFloatArrayRealSize(unsigned short aSamplesNumber, unsigned short aFormatCode) {
    int size;
    if (aFormatCode == 1 || aFormatCode == 2 || aFormatCode == 4 || aFormatCode == 5) {
        size = 4 * aSamplesNumber;
    } else if (aFormatCode == 3) {
        size = 2 * aSamplesNumber;
    } else if (aFormatCode == 8) {
        size = aSamplesNumber;
    } else {
        throw exceptions::UnsupportedFeatureException();
    }
    return size;
}

int FloatingPointFormatter::Format(char *apSrc, size_t aSrcSize, size_t aSamplesNumber, short aFormat) {
    int rc;
    switch (aFormat) {
        case 1:
            /// Convert IBM float to native floats.
            rc = FloatingPointFormatter::FromIBM((unsigned char *&) apSrc, aSrcSize, aSamplesNumber);
            break;
        case 2:
            /// Convert 4 byte two's complement integer to native floats.
            rc = FloatingPointFormatter::FromLong((unsigned char *&) apSrc, aSrcSize, aSamplesNumber);
            break;
        case 3:
            /// Convert 2 byte two's complement integer to native floats.
            rc = FloatingPointFormatter::FromShort((unsigned char *&) apSrc, aSrcSize, aSamplesNumber);
            break;
        case 5:
            /// Convert IEEE float to native floats.
            rc = FloatingPointFormatter::FromIEEE((unsigned char *&) apSrc, aSrcSize, aSamplesNumber);
            break;
        case 8:
            /// Convert 1 byte two's complement integer  to native floats.
            rc = FloatingPointFormatter::FromChar((unsigned char *&) apSrc, aSrcSize, aSamplesNumber);
            break;
        default:
            throw exceptions::UnsupportedFeatureException();
    }
    return rc;
}

int FloatingPointFormatter::FromIBM(unsigned char *&apSrc, size_t aSrcSize, size_t aSamplesNumber) {
    bool is_little_endian = Checker::IsLittleEndianMachine();
    size_t size_entry = sizeof(float);
    size_t size = aSrcSize / size_entry;
    unsigned int fconv, fmant, t;
    unsigned int offset = 0;
    for (int i = 0; i < size; ++i) {
        offset = i * size_entry;
        memcpy(&fconv, apSrc + offset, sizeof(unsigned int));
        if (is_little_endian) {
            fconv = (fconv << 24) | ((fconv >> 24) & 0xff) | ((fconv & 0xff00) << 8) |
                    ((fconv & 0xff0000) >> 8);
        }
        if (fconv) {
            fmant = 0x00ffffff & fconv;
            t = (int) ((0x7f000000 & fconv) >> 22) - 130;
            while (!(fmant & 0x00800000)) {
                --t;
                fmant <<= 1;
            }
            if (t > 254) {
                fconv = (0x80000000 & fconv) | 0x7f7fffff;
            } else if (t <= 0) {
                fconv = 0;
            } else {
                fconv = (0x80000000 & fconv) | (t << 23) | (0x007fffff & fmant);
            }
        }
        memcpy(apSrc + offset, &fconv, sizeof(float));
    }
    return 1;
}

int FloatingPointFormatter::FromLong(unsigned char *&apSrc, size_t aSize, size_t aSamplesNumber) {
    return 1;
}

int FloatingPointFormatter::FromShort(unsigned char *&apSrc, size_t aSize, size_t aSamplesNumber) {
    return 1;
}

int FloatingPointFormatter::FromIEEE(unsigned char *&apSrc, size_t aSize, size_t aSamplesNumber) {
    return 1;
}

int FloatingPointFormatter::FromChar(unsigned char *&apSrc, size_t aSize, size_t aSamplesNumber) {
    return 1;
}
