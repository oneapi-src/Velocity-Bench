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
// Created by pancee on 11/4/20.
//

#ifndef THOTH_DATA_UNITS_DATA_TYPES_TRACE_HEADER_HPP
#define THOTH_DATA_UNITS_DATA_TYPES_TRACE_HEADER_HPP

#include <string>
#include <unordered_map>
#include <map>
#include <cctype>
#include <cstring>
#include <cstdio>

namespace thoth {
    namespace dataunits {
        /**
         * @brief Trace Header
         */
        class TraceHeaderKey {
        public:
            /**
            * @brief Enum with the different Trace Headers.
            */
            enum Key : char {
                TRACL,
                TRACR,
                FLDR,
                TRACF,
                EP,
                CDP,
                CDPT,
                TRID,
                NVS,
                NHS,
                DUSE,
                OFFSET,
                GELEV,
                SELEV,
                SDEPTH,
                GDEL,
                SDEL,
                SWDEP,
                GWDEP,
                SCALEL,
                SCALCO,
                SX,
                SY,
                GX,
                GY,
                COINTIT,
                WEVEL,
                SWEVEL,
                SUT,
                GUT,
                SSTAT,
                GSTAT,
                TSTAT,
                LAGA,
                LAGB,
                DELRT,
                MUTS,
                MUTE,
                DT,
                GAIN,
                IGC,
                IGI,
                CORR,
                SFS,
                SFE,
                SLEN,
                STYP,
                STAS,
                STAE,
                TATYP,
                AFILF,
                AFILS,
                NOFILF,
                NOFILS,
                LCF,
                HCF,
                LCS,
                HCS,
                YEAR,
                DAY,
                HOUR,
                MINUTE,
                SEC,
                TIMBAS,
                TRWF,
                GRNORS,
                GRNOFR,
                GRNLOF,
                GAPS,
                OTRAV,
                D1,
                F1,
                D2,
                F2,
                UNGPOW,
                UNSCALE,
                NTR,
                MARK,
                SHORTPAD,
                NS
            };

        public:
            /**
             * @brief Trace Header constructor.
             */
            TraceHeaderKey(Key aKey) {
                this->mKey = aKey;
            }

            /**
             * @brief Copy constructor, to copy char  representing one of the enum to the class.
             *
             * @param[in] aType
             * The char mValue representing one of the enums.
             */
            explicit constexpr TraceHeaderKey(char aValue)
                    : mKey(static_cast<TraceHeaderKey::Key>(aValue)) {}

            /**
             * @brief Trace Header Destructor.
             */
            ~TraceHeaderKey() = default;


            /**
             * @brief Equality operator overloading.
             *
             * @param[in] aType
             * Comparison target.
             *
             * @return True if they contain the same type.
             */
            constexpr bool operator==(TraceHeaderKey aTraceHeaderAttribute) const {
                return this->mKey == aTraceHeaderAttribute.mKey;
            }

            /**
             * @brief Inequality operator overloading.
             *
             * @param[in] aTraceHeaderAttribute
             * Comparison target.
             *
             * @return True if they do not contain the same type.
             */
            constexpr bool operator!=(TraceHeaderKey aTraceHeaderAttribute) const {
                return this->mKey != aTraceHeaderAttribute.mKey;
            }

            /**
             * @brief Greater equality operator overloading.
             *
             * @param[in] aTraceHeaderAttribute
             * Comparison target.
             *
             * @return True if aTraceHeaderAttribute is greater than current object.
             */
            constexpr bool operator>(TraceHeaderKey aTraceHeaderAttribute) const {
                return this->mKey > aTraceHeaderAttribute.mKey;
            }

            /**
             * @brief Smaller equality operator overloading.
             *
             * @param[in] aTraceHeaderAttribute
             * Comparison target.
             *
             * @return True if aTraceHeaderAttribute is smaller than current object.
             */
            constexpr bool operator<(TraceHeaderKey aTraceHeaderAttribute) const {
                return this->mKey < aTraceHeaderAttribute.mKey;
            }

            /**
             * @brief Allow switch and comparisons.
             * @return The enum mValue.
             */
            explicit operator Key() const { return this->mKey; }  // Allow switch and comparisons.

            /**
             * @brief Prevent usage of if (aTraceHeaderAttribute)
             * @return Nothing as it's deleted.
             */
            explicit operator bool() = delete;


            /**
             * @brief Convert trace header to string.
             * @return A string representation of the trace header.
             */
            std::string ToString() {
                static const std::unordered_map<TraceHeaderKey::Key, std::string> ENUM_TO_STRING = {
                        {TRACL,    "TRACL"},
                        {TRACR,    "TRACR"},
                        {FLDR,     "FLDR"},
                        {TRACF,    "TRACF"},
                        {EP,       "EP"},
                        {CDP,      "CDP"},
                        {CDPT,     "CDPT"},
                        {TRID,     "TRID"},
                        {NVS,      "NVS"},
                        {NHS,      "NHS"},
                        {DUSE,     "DUSE"},
                        {OFFSET,   "Offset"},
                        {GELEV,    "GELEV"},
                        {SELEV,    "SELEV"},
                        {SDEPTH,   "SDEPTH"},
                        {GDEL,     "GDEL"},
                        {SDEL,     "SDEL"},
                        {SWDEP,    "SWDEP"},
                        {GWDEP,    "DWDEP"},
                        {SCALEL,   "SCALEL"},
                        {SCALCO,   "SCALCO"},
                        {SX,       "SX"},
                        {SY,       "SY"},
                        {GX,       "GX"},
                        {GY,       "GY"},
                        {COINTIT,  "COINTIT"},
                        {WEVEL,    "WEVEL"},
                        {SWEVEL,   "SWEVEL"},
                        {SUT,      "SUT"},
                        {GUT,      "GUT"},
                        {SSTAT,    "SSTAT"},
                        {GSTAT,    "GSTAT"},
                        {TSTAT,    "TSTAT"},
                        {LAGA,     "LAGA"},
                        {LAGB,     "LAGB"},
                        {DELRT,    "DELRT"},
                        {MUTS,     "MUTS"},
                        {MUTE,     "MUTE"},
                        {DT,       "DT"},
                        {GAIN,     "GAIN"},
                        {IGC,      "IGC"},
                        {IGI,      "IGI"},
                        {CORR,     "CORR"},
                        {SFS,      "SFS"},
                        {SFE,      "SFE"},
                        {SLEN,     "SLEN"},
                        {STYP,     "STYP"},
                        {STAS,     "STAS"},
                        {STAE,     "STAE"},
                        {TATYP,    "TATYP"},
                        {AFILF,    "AFILF"},
                        {AFILS,    "AFILS"},
                        {NOFILF,   "NOFILF"},
                        {NOFILS,   "NOFILS"},
                        {LCF,      "LCF"},
                        {HCF,      "HCF"},
                        {LCS,      "LCS"},
                        {HCS,      "HCS"},
                        {YEAR,     "YEAR"},
                        {DAY,      "DAY"},
                        {HOUR,     "HOUR"},
                        {MINUTE,   "MINUTE"},
                        {SEC,      "SEC"},
                        {TIMBAS,   "TIMBAS"},
                        {TRWF,     "TRWF"},
                        {GRNORS,   "GRNORS"},
                        {GRNOFR,   "GRNOFR"},
                        {GRNLOF,   "GRNLOF"},
                        {GAPS,     "GAPS"},
                        {OTRAV,    "OTRAV"},
                        {D1,       "D1"},
                        {F1,       "F1"},
                        {D2,       "D2"},
                        {F2,       "F2"},
                        {UNGPOW,   "UNGPOW"},
                        {UNSCALE,  "UNSCALE"},
                        {NTR,      "NTR"},
                        {MARK,     "MARK"},
                        {SHORTPAD, "SHORTPAD"},
                        {NS,       "NS"}
                };
                return ENUM_TO_STRING.find(this->mKey)->second;
            }

            /**
             * @brief Convert trace header type to char.
             * @return A char representation of the trace header.
             */
            inline char ToChar() const { return this->mKey; }

            /**
             * @brief Provides the hash for TraceHeaderKey object.
             *
             * @return A hash representation of this object.
             */
            size_t hash() const {
                return std::hash<Key>()(this->mKey);
            }

        private:
            /// @brief Key for trace header.
            /// @see {TraceHeaderKey::Key}
            Key mKey;
        };
    }//namespace dataunits
} //namespace thoth

namespace std {
    /**
     * @brief The hash function used by the STL containers for TraceHeaderKey class.
     */
    template<>
    struct hash<thoth::dataunits::TraceHeaderKey> {
        std::size_t operator()(const thoth::dataunits::TraceHeaderKey &f) const {
            return f.hash();
        }
    };
} //namespace std

#endif //THOTH_DATA_UNITS_DATA_TYPES_TRACE_HEADER_HPP
