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


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   segy.h
 * Author: wsl
 *
 * Created on May 22, 2017, 5:22 PM
 */

#ifndef SUSEGY_HELPER
#define SUSEGY_HELPER

#include <bitset>
#include <endian.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "suheaders.h"
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include "segyelement.h"

static const unsigned char a2e[256] = {
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, /*         	*/
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, /*         	*/
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, /*         	*/
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, /*         	*/
        0x40, 0x4F, 0x7F, 0x7B, 0x5B, 0x6C, 0x50, 0x7D, /*  !"#$%&'	*/
        0x4D, 0x5D, 0x5C, 0x4E, 0x6B, 0x60, 0x4B, 0x61, /* ()*+,-./	*/
        0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, /* 01234567	*/
        0xF8, 0xF9, 0x7A, 0x5E, 0x4C, 0x7E, 0x6E, 0x6F, /* 89:;<=>?	*/
        0x7C, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, /* @ABCDEFG	*/
        0xC8, 0xC9, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, /* HIJKLMNO	*/
        0xD7, 0xD8, 0xD9, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, /* PQRSTUVW	*/
        0xE7, 0xE8, 0xE9, 0x4A, 0xE0, 0x5A, 0x5F, 0x6D, /* XYZ[\]^_	*/
        0x79, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, /* `abcdefg	*/
        0x88, 0x89, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, /* hijklmno	*/
        0x97, 0x98, 0x99, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, /* pqrstuvw	*/
        0xA7, 0xA8, 0xA9, 0xC0, 0x6A, 0xD0, 0xA1, 0x40, /* xyz{|}~ 	*/
        0xB9, 0xBA, 0xED, 0xBF, 0xBC, 0xBD, 0xEC, 0xFA, /*              */
        0xCB, 0xCC, 0xCD, 0xCE, 0xCF, 0xDA, 0xDB, 0xDC, /*             	*/
        0xDE, 0xDF, 0xEA, 0xEB, 0xBE, 0xCA, 0xBB, 0xFE, /*		*/
        0xFB, 0xFD, 0x7d, 0xEF, 0xEE, 0xFC, 0xB8, 0xDD, /*		*/
        0x77, 0x78, 0xAF, 0x8D, 0x8A, 0x8B, 0xAE, 0xB2, /*		*/
        0x8F, 0x90, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F, /*		*/
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, /*	       	*/
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, /*	       	*/
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, /*	       	*/
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, /*	       	*/
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, /*	       	*/
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, /*	       	*/
        0xAA, 0xAB, 0xAC, 0xAD, 0x8C, 0x8E, 0x80, 0xB6, /* ����	*/
        0xB3, 0xB5, 0xB7, 0xB1, 0xB0, 0xB4, 0x76, 0xA0, /* 	*/
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, /*        	*/
        0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40  /*        	*/
};
static const unsigned char e2a[256] = {
        0, 1, 2, 3, 156, 9, 134, 127, 151, 141, 142, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 157, 133, 8, 135, 24, 25, 146, 143, 28, 29,
        30, 31, 128, 129, 130, 131, 132, 10, 23, 27, 136, 137, 138, 139, 140,
        5, 6, 7, 144, 145, 22, 147, 148, 149, 150, 4, 152, 153, 154, 155,
        20, 21, 158, 26, 32, 160, 161, 162, 163, 164, 165, 166, 167, 168, 91,
        46, 60, 40, 43, 33, 38, 169, 170, 171, 172, 173, 174, 175, 176, 177,
        93, 36, 42, 41, 59, 94, 45, 47, 178, 179, 180, 181, 182, 183, 184,
        185, 124, 44, 37, 95, 62, 63, 186, 187, 188, 189, 190, 191, 192, 193,
        194, 96, 58, 35, 64, 39, 61, 34, 195, 97, 98, 99, 100, 101, 102,
        103, 104, 105, 196, 197, 198, 199, 200, 201, 202, 106, 107, 108, 109, 110,
        111, 112, 113, 114, 203, 204, 205, 206, 207, 208, 209, 126, 115, 116, 117,
        118, 119, 120, 121, 122, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
        220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 123, 65, 66,
        67, 68, 69, 70, 71, 72, 73, 232, 233, 234, 235, 236, 237, 125, 74,
        75, 76, 77, 78, 79, 80, 81, 82, 238, 239, 240, 241, 242, 243, 92,
        159, 83, 84, 85, 86, 87, 88, 89, 90, 244, 245, 246, 247, 248, 249,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 250, 251, 252, 253, 254,
        255};

void ebcdicToAscii(unsigned char *s, unsigned char *ascii);

void ebcdicToAscii(unsigned char *s);

void ibm_to_float(int from[], int to[], int n, int endian);

////void float_to_ibm(int from[], int to[], int n, int endian);

void long_to_float(long from[], float to[], int n, int endian);

void short_to_float(short from[], float to[], int n, int endian);

void integer1_to_float(signed char from[], float to[], int n);

void ieee2ibm(void *to, const void *from, int len);

void tapebhed_to_bhed(const tapebhed *tapebhptr, bhed *bhptr);

void tapesegy_to_segy(const tapesegy *tapetrptr, segy *trptr);

void segy_to_tapesegy(segy *trptr, tapesegy *tapetrptr, int nsegy);

void bhed_to_tapebhed(bhed *bhptr, tapebhed *tapebhptr);

#endif // SUSEGY_HELPER
