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

#include <thoth/utils/convertors/NumbersConvertor.hpp>

using namespace thoth::utils::convertors;


char *NumbersConvertor::ToLittleEndian(char *apSrc, size_t aSize, short aFormat) {
    int rc = 0;
    switch (aFormat) {
        case 1:
            /// Convert IBM float to native floats.
            NumbersConvertor::ToLittleEndian((int *) apSrc, aSize);
            break;
        case 2:
            /// Convert 4 byte two's complement integer to native floats.
//            NumbersConvertor::ToLittleEndian((long *) apSrc, aSize);
            break;
        case 3:
            /// Convert 2 byte two's complement integer to native floats.
            NumbersConvertor::ToLittleEndian((short *) apSrc, aSize);
            break;
        case 5:
            /// Convert IEEE float to native floats.
            NumbersConvertor::ToLittleEndian((short *) apSrc, aSize);
            break;
        case 8:
            /// Convert 1 byte two's complement integer  to native floats.
            NumbersConvertor::ToLittleEndian((signed char *) apSrc, aSize);
            break;
    }
    return apSrc;
}

short int *NumbersConvertor::ToLittleEndian(short int *apSrc, size_t aSize) {
    for (int i = 0; i < aSize; ++i) {
        apSrc[i] = ToLittleEndian(apSrc[i]);
    }
    return apSrc;
}

short int NumbersConvertor::ToLittleEndian(short int aSrc) {
    short int tmp = aSrc >> 8;
    return (aSrc << 8) | (tmp);
}

unsigned short int *NumbersConvertor::ToLittleEndian(unsigned short int *apSrc, size_t aSize) {
    for (int i = 0; i < aSize; ++i) {
        apSrc[i] = ToLittleEndian(apSrc[i]);
    }
    return apSrc;
}

unsigned short int NumbersConvertor::ToLittleEndian(unsigned short int aSrc) {
    unsigned short int tmp = aSrc >> 8;
    return (aSrc << 8) | (tmp);
}

int *NumbersConvertor::ToLittleEndian(int *apSrc, size_t aSize) {
    for (int i = 0; i < aSize; ++i) {
        apSrc[i] = ToLittleEndian(apSrc[i]);
    }
    return apSrc;
}

int NumbersConvertor::ToLittleEndian(int aSrc) {
    unsigned short int tmp1 = (aSrc >> 16);
    unsigned short int tmp2 = (aSrc & 0x0000FFFF);
    tmp2 = NumbersConvertor::ToLittleEndian(tmp2);
    tmp1 = NumbersConvertor::ToLittleEndian(tmp1);

    int aDst = (int) tmp2;
    aDst = aDst << 16;
    aDst = aDst | (int) tmp1;

    return aDst;
}

unsigned int *NumbersConvertor::ToLittleEndian(unsigned int *apSrc, size_t aSize) {
    for (int i = 0; i < aSize; ++i) {
        apSrc[i] = ToLittleEndian(apSrc[i]);
    }
    return apSrc;
}

unsigned int NumbersConvertor::ToLittleEndian(unsigned int aSrc) {
    unsigned short int tmp1 = (aSrc >> 16);
    unsigned short int tmp2 = (aSrc & 0x0000FFFF);
    tmp2 = NumbersConvertor::ToLittleEndian(tmp2);
    tmp1 = NumbersConvertor::ToLittleEndian(tmp1);

    auto aDst = (unsigned int) tmp2;
    aDst = aDst << 16;
    aDst = aDst | (unsigned int) tmp1;

    return aDst;
}

signed char *NumbersConvertor::ToLittleEndian(signed char *apSrc, size_t aSize) {
    for (int i = 0; i < aSize; ++i) {
        apSrc[i] = ToLittleEndian(apSrc[i]);
    }
    return apSrc;
}

signed char NumbersConvertor::ToLittleEndian(signed char aSrc) {
    char tmp = aSrc >> 8;
    return (aSrc << 8) | (tmp);
}
