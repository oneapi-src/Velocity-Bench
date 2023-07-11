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

#ifndef THOTH_LOOKUPS_TRACE_HEADERS_LOOKUP_HPP
#define THOTH_LOOKUPS_TRACE_HEADERS_LOOKUP_HPP

namespace thoth {
    namespace lookups {

#define IO_SIZE_TRACE_HEADER        240      /* Size of the trace header in SEG-Y and SU files */
#define IO_POS_S_TRACE_HEADER       3600     /* Start position of trace header in SEG-Y and SU files */

        /**
         * @brief SEG-Y and SU trace header lookup. Variable types corresponds to
         * the number of allocated byte(s) for this variable internally in the SEG-Y or SU file
         * according to the general format.
         *
         * @note All variables are stored as big endian, if the machine used is little endian
         * big endian to little endian conversion should take place.
         */
        struct TraceHeaderLookup {
            int32_t TRACL;
            int32_t TRACR;
            int32_t FLDR;
            int32_t TRACF;
            int32_t EP;
            int32_t CDP;
            int32_t CDPT;
            int16_t TRACE_CODE;
            int16_t NVS;
            int16_t NHS;
            int16_t DUSE;
            int32_t OFFSET;
            int32_t GELEV;
            int32_t SELEV;
            int32_t SDEPTH;
            int32_t GDEL;
            int32_t SDEL;
            int32_t SWDEP;
            int32_t GWDEP;
            int16_t SCALEL;
            int16_t SCALCO;
            int32_t SX;
            int32_t SY;
            int32_t GX;
            int32_t GY;
            int16_t COINTIT;
            int16_t WEVEL;
            int16_t SWEVEL;
            int16_t SUT;
            int16_t GUT;
            int16_t SSTAT;
            int16_t GSTAT;
            int16_t TSTAT;
            int16_t LAGA;
            int16_t LAGB;
            int16_t DELRT;
            int16_t MUTS;
            int16_t MUTE;
            uint16_t NS;
            uint16_t DT;
            int16_t GAIN;
            int16_t IGC;
            int16_t IGI;
            int16_t CORR;
            int16_t SFS;
            int16_t SFE;
            int16_t SLEN;
            int16_t STYP;
            int16_t STAS;
            int16_t STAE;
            int16_t TATYP;
            int16_t AFILF;
            int16_t AFILS;
            int16_t NOFILF;
            int16_t NOFILS;
            int16_t LCF;
            int16_t HCF;
            int16_t LCS;
            int16_t HCS;
            int16_t YEAR;
            int16_t DAY;
            int16_t HOUR;
            int16_t MINUTE;
            int16_t SEC;
            int16_t TIMBAS;
            int16_t TRWF;
            int16_t GRNORS;
            int16_t GRNOFR;
            int16_t GRNLOF;
            int16_t GAPS;
            int16_t OTRAV;

            /*
             * Extended.
             */

            int32_t ENSX;
            int32_t ENSY;
            int32_t INLINE;
            int32_t CROSS;
            int32_t SHOOTPOINT;
            int16_t SHOOTPOINT_SCALE;
            int16_t TRACE_UNIT;
            uint8_t TRANSD_CONST[6];
            int16_t TRANSD_UNIT;
            int16_t TRID;
            int16_t SCALE_TIME;
            int16_t SRC_ORIENT;
            uint8_t SRC_DIRECTION[6];
            uint8_t SRC_MEASUREMT[6];
            int16_t SRC_UNIT;
            uint8_t UNASSIGNED1[6];
        };
    } //namespace lookups
} //namespace thoth

#endif //THOTH_LOOKUPS_TRACE_HEADERS_LOOKUP_HPP
