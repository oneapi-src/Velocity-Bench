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
// Created by zeyad-osama on 21/01/2021.
//

#include <thoth/lookups/SeismicFilesHeaders.hpp>

using namespace thoth::lookups;
using namespace thoth::dataunits;
using namespace thoth::utils::range;


const std::map<TraceHeaderKey, ByteRange> SeismicFilesHeaders::mTraceHeadersMap = {
        {TraceHeaderKey(TraceHeaderKey::TRACL),   ByteRange('P', 0)},
        {TraceHeaderKey(TraceHeaderKey::TRACR),   ByteRange('P', 4)},
        {TraceHeaderKey(TraceHeaderKey::FLDR),    ByteRange('P', 8)},
        {TraceHeaderKey(TraceHeaderKey::TRACF),   ByteRange('P', 12)},
        {TraceHeaderKey(TraceHeaderKey::EP),      ByteRange('P', 16)},
        {TraceHeaderKey(TraceHeaderKey::CDP),     ByteRange('P', 20)},
        {TraceHeaderKey(TraceHeaderKey::CDPT),    ByteRange('P', 24)},
        {TraceHeaderKey(TraceHeaderKey::TRID),    ByteRange('U', 28)},
        {TraceHeaderKey(TraceHeaderKey::NVS),     ByteRange('U', 30)},
        {TraceHeaderKey(TraceHeaderKey::NHS),     ByteRange('U', 32)},
        {TraceHeaderKey(TraceHeaderKey::DUSE),    ByteRange('U', 34)},
        {TraceHeaderKey(TraceHeaderKey::OFFSET),  ByteRange('P', 36)},
        {TraceHeaderKey(TraceHeaderKey::GELEV),   ByteRange('P', 40)},
        {TraceHeaderKey(TraceHeaderKey::SELEV),   ByteRange('P', 44)},
        {TraceHeaderKey(TraceHeaderKey::SDEPTH),  ByteRange('P', 48)},
        {TraceHeaderKey(TraceHeaderKey::GDEL),    ByteRange('P', 52)},
        {TraceHeaderKey(TraceHeaderKey::SDEL),    ByteRange('P', 56)},
        {TraceHeaderKey(TraceHeaderKey::SWDEP),   ByteRange('P', 60)},
        {TraceHeaderKey(TraceHeaderKey::GWDEP),   ByteRange('P', 64)},
        {TraceHeaderKey(TraceHeaderKey::SCALEL),  ByteRange('U', 68)},
        {TraceHeaderKey(TraceHeaderKey::SCALCO),  ByteRange('U', 70)},
        {TraceHeaderKey(TraceHeaderKey::SX),      ByteRange('P', 72)},
        {TraceHeaderKey(TraceHeaderKey::SY),      ByteRange('P', 76)},
        {TraceHeaderKey(TraceHeaderKey::GX),      ByteRange('P', 80)},
        {TraceHeaderKey(TraceHeaderKey::GY),      ByteRange('P', 84)},
        {TraceHeaderKey(TraceHeaderKey::COINTIT), ByteRange('U', 88)},
        {TraceHeaderKey(TraceHeaderKey::WEVEL),   ByteRange('U', 90)},
        {TraceHeaderKey(TraceHeaderKey::SWEVEL),  ByteRange('U', 92)},
        {TraceHeaderKey(TraceHeaderKey::SUT),     ByteRange('U', 94)},
        {TraceHeaderKey(TraceHeaderKey::GUT),     ByteRange('U', 96)},
        {TraceHeaderKey(TraceHeaderKey::SSTAT),   ByteRange('U', 98)},
        {TraceHeaderKey(TraceHeaderKey::GSTAT),   ByteRange('U', 100)},
        {TraceHeaderKey(TraceHeaderKey::TSTAT),   ByteRange('U', 102)},
        {TraceHeaderKey(TraceHeaderKey::LAGA),    ByteRange('U', 104)},
        {TraceHeaderKey(TraceHeaderKey::LAGB),    ByteRange('U', 106)},
        {TraceHeaderKey(TraceHeaderKey::DELRT),   ByteRange('U', 108)},
        {TraceHeaderKey(TraceHeaderKey::MUTS),    ByteRange('U', 110)},
        {TraceHeaderKey(TraceHeaderKey::MUTE),    ByteRange('U', 112)},
        {TraceHeaderKey(TraceHeaderKey::NS),      ByteRange('U', 114)},
        {TraceHeaderKey(TraceHeaderKey::DT),      ByteRange('U', 116)},
        {TraceHeaderKey(TraceHeaderKey::GAIN),    ByteRange('U', 118)},
        {TraceHeaderKey(TraceHeaderKey::IGC),     ByteRange('U', 120)},
        {TraceHeaderKey(TraceHeaderKey::IGI),     ByteRange('U', 122)},
        {TraceHeaderKey(TraceHeaderKey::CORR),    ByteRange('U', 124)},
        {TraceHeaderKey(TraceHeaderKey::SFS),     ByteRange('U', 126)},
        {TraceHeaderKey(TraceHeaderKey::SFE),     ByteRange('U', 128)},
        {TraceHeaderKey(TraceHeaderKey::SLEN),    ByteRange('U', 130)},
        {TraceHeaderKey(TraceHeaderKey::STYP),    ByteRange('U', 132)},
        {TraceHeaderKey(TraceHeaderKey::STAS),    ByteRange('U', 134)},
        {TraceHeaderKey(TraceHeaderKey::STAE),    ByteRange('U', 136)},
        {TraceHeaderKey(TraceHeaderKey::TATYP),   ByteRange('U', 138)},
        {TraceHeaderKey(TraceHeaderKey::AFILF),   ByteRange('U', 140)},
        {TraceHeaderKey(TraceHeaderKey::AFILS),   ByteRange('U', 142)},
        {TraceHeaderKey(TraceHeaderKey::NOFILF),  ByteRange('U', 144)},
        {TraceHeaderKey(TraceHeaderKey::NOFILS),  ByteRange('U', 146)},
        {TraceHeaderKey(TraceHeaderKey::LCF),     ByteRange('U', 148)},
        {TraceHeaderKey(TraceHeaderKey::HCF),     ByteRange('U', 150)},
        {TraceHeaderKey(TraceHeaderKey::LCS),     ByteRange('U', 152)},
        {TraceHeaderKey(TraceHeaderKey::HCS),     ByteRange('U', 154)},
        {TraceHeaderKey(TraceHeaderKey::YEAR),    ByteRange('U', 156)},
        {TraceHeaderKey(TraceHeaderKey::DAY),     ByteRange('U', 158)},
        {TraceHeaderKey(TraceHeaderKey::HOUR),    ByteRange('U', 160)},
        {TraceHeaderKey(TraceHeaderKey::MINUTE),  ByteRange('U', 162)},
        {TraceHeaderKey(TraceHeaderKey::SEC),     ByteRange('U', 164)},
        {TraceHeaderKey(TraceHeaderKey::TIMBAS),  ByteRange('U', 166)},
        {TraceHeaderKey(TraceHeaderKey::TRWF),    ByteRange('U', 168)},
        {TraceHeaderKey(TraceHeaderKey::GRNORS),  ByteRange('U', 170)},
        {TraceHeaderKey(TraceHeaderKey::GRNOFR),  ByteRange('U', 172)},
        {TraceHeaderKey(TraceHeaderKey::GRNLOF),  ByteRange('U', 174)},
        {TraceHeaderKey(TraceHeaderKey::GAPS),    ByteRange('U', 176)},
        {TraceHeaderKey(TraceHeaderKey::OTRAV),   ByteRange('U', 178)}
};

