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
// Created by zeyad-osama on 07/03/2021.
//

#ifndef THOTH_LOOKUPS_BINARY_HEADERS_LOOKUP_HPP
#define THOTH_LOOKUPS_BINARY_HEADERS_LOOKUP_HPP

namespace thoth {
    namespace lookups {

#define IO_SIZE_BINARY_HEADER       400      /* Size of the binary header in SEG-Y and SU files */
#define IO_POS_S_BINARY_HEADER      3200     /* Start position of binary header in SEG-Y and SU files */
#define IO_POS_E_BINARY_HEADER      3600     /* End position of binary header in SEG-Y and SU files */

        /**
         * @brief SEG-Y and SU binary header lookup. Variable types corresponds to
         * the number of allocated byte(s) for this variable internally in the SEG-Y or SU file
         * according to the general format.
         *
         * @note All variables are stored as big endian, if the machine used is little endian
         * big endian to little endian conversion should take place.
         */
        struct BinaryHeaderLookup {
            int32_t JOBID;
            int32_t LINO;
            int32_t RENO;
            int16_t NTRPR;
            int16_t NART;
            int16_t HDT;
            int16_t DTO;
            uint16_t HNS;
            uint16_t NSO;
            int16_t FORMAT;
            int16_t FOLD;
            int16_t TSORT;
            int16_t VSCODE;
            int16_t HSFS;
            int16_t HSFE;
            int16_t HSLEN;
            int16_t HSTYP;
            int16_t SCHN;
            int16_t HSTAS;
            int16_t HSTAE;
            int16_t HTATYP;
            int16_t HCORR;
            int16_t BGRCV;
            int16_t RCVM;
            int16_t MFEET;
            int16_t POLYT;
            int16_t VPOL;
            uint8_t UNASSIGNED1[240];
            int16_t SEGY_REV_NUM;
            int16_t FIXED_LEN;
            int16_t EXT_HEAD;
            uint8_t UNASSIGNED2[94];
        };
    } //namespace lookups
} //namespace thoth

#endif //THOTH_LOOKUPS_BINARY_HEADERS_LOOKUP_HPP
