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


////
//// Created by zeyad-osama on 12/03/2021.
////
//
//#include <thoth/lookups/mappers/TraceHeaderMapper.hpp>
//
//#include <thoth/exceptions/Exceptions.hpp>
//
//using namespace thoth::lookups;
//using namespace thoth::dataunits;
//using namespace thoth::exceptions;
//
//
//size_t TraceHeaderMapper::GetTraceHeaderValue(const TraceHeaderKey::Key &aTraceHeaderKey,
//                                              const TraceHeaderLookup &aTraceHeaderLookup) {
//    size_t val;
//    switch (aTraceHeaderKey) {
//        case TraceHeaderKey::TRACL:
//            val = aTraceHeaderLookup.TRACL;
//            break;
//        case TraceHeaderKey::TRACR:
//            val = aTraceHeaderLookup.TRACR;
//            break;
//        case TraceHeaderKey::FLDR:
//            val = aTraceHeaderLookup.FLDR;
//            break;
//        case TraceHeaderKey::TRACF:
//            val = aTraceHeaderLookup.TRACF;
//            break;
//        case TraceHeaderKey::EP:
//            val = aTraceHeaderLookup.EP;
//            break;
//        case TraceHeaderKey::CDP:
//            val = aTraceHeaderLookup.CDP;
//            break;
//        case TraceHeaderKey::CDPT:
//            val = aTraceHeaderLookup.CDPT;
//            break;
//        case TraceHeaderKey::TRID:
//            val = aTraceHeaderLookup.TRID;
//            break;
//        case TraceHeaderKey::NVS:
//            val = aTraceHeaderLookup.NVS;
//            break;
//        case TraceHeaderKey::NHS:
//            val = aTraceHeaderLookup.NHS;
//            break;
//        case TraceHeaderKey::DUSE:
//            val = aTraceHeaderLookup.DUSE;
//            break;
//        case TraceHeaderKey::OFFSET:
//            val = aTraceHeaderLookup.OFFSET;
//            break;
//        case TraceHeaderKey::GELEV:
//            val = aTraceHeaderLookup.GELEV;
//            break;
//        case TraceHeaderKey::SELEV:
//            val = aTraceHeaderLookup.SELEV;
//            break;
//        case TraceHeaderKey::SDEPTH:
//            val = aTraceHeaderLookup.SDEPTH;
//            break;
//        case TraceHeaderKey::GDEL:
//            val = aTraceHeaderLookup.GDEL;
//            break;
//        case TraceHeaderKey::SDEL:
//            val = aTraceHeaderLookup.SDEL;
//            break;
//        case TraceHeaderKey::SWDEP:
//            val = aTraceHeaderLookup.SWDEP;
//            break;
//        case TraceHeaderKey::GWDEP:
//            val = aTraceHeaderLookup.GWDEP;
//            break;
//        case TraceHeaderKey::SCALEL:
//            val = aTraceHeaderLookup.SCALEL;
//            break;
//        case TraceHeaderKey::SCALCO:
//            val = aTraceHeaderLookup.SCALCO;
//            break;
//        case TraceHeaderKey::SX:
//            val = aTraceHeaderLookup.SX;
//            break;
//        case TraceHeaderKey::SY:
//            val = aTraceHeaderLookup.SY;
//            break;
//        case TraceHeaderKey::GX:
//            val = aTraceHeaderLookup.GX;
//            break;
//        case TraceHeaderKey::GY:
//            val = aTraceHeaderLookup.GY;
//            break;
//        case TraceHeaderKey::COINTIT:
//            val = aTraceHeaderLookup.COINTIT;
//            break;
//        case TraceHeaderKey::WEVEL:
//            val = aTraceHeaderLookup.WEVEL;
//            break;
//        case TraceHeaderKey::SWEVEL:
//            val = aTraceHeaderLookup.SWEVEL;
//            break;
//        case TraceHeaderKey::SUT:
//            val = aTraceHeaderLookup.SUT;
//            break;
//        case TraceHeaderKey::GUT:
//            val = aTraceHeaderLookup.GUT;
//            break;
//        case TraceHeaderKey::SSTAT:
//            val = aTraceHeaderLookup.SSTAT;
//            break;
//        case TraceHeaderKey::GSTAT:
//            val = aTraceHeaderLookup.GSTAT;
//            break;
//        case TraceHeaderKey::TSTAT:
//            val = aTraceHeaderLookup.TSTAT;
//            break;
//        case TraceHeaderKey::LAGA:
//            val = aTraceHeaderLookup.LAGA;
//            break;
//        case TraceHeaderKey::LAGB:
//            val = aTraceHeaderLookup.LAGB;
//            break;
//        case TraceHeaderKey::DELRT:
//            val = aTraceHeaderLookup.DELRT;
//            break;
//        case TraceHeaderKey::MUTS:
//            val = aTraceHeaderLookup.MUTS;
//            break;
//        case TraceHeaderKey::MUTE:
//            val = aTraceHeaderLookup.MUTE;
//            break;
//        case TraceHeaderKey::DT:
//            val = aTraceHeaderLookup.DT;
//            break;
//        case TraceHeaderKey::GAIN:
//            val = aTraceHeaderLookup.GAIN;
//            break;
//        case TraceHeaderKey::IGC:
//            val = aTraceHeaderLookup.IGC;
//            break;
//        case TraceHeaderKey::IGI:
//            val = aTraceHeaderLookup.IGI;
//            break;
//        case TraceHeaderKey::CORR:
//            val = aTraceHeaderLookup.CORR;
//            break;
//        case TraceHeaderKey::SFS:
//            val = aTraceHeaderLookup.SFS;
//            break;
//        case TraceHeaderKey::SFE:
//            val = aTraceHeaderLookup.SFE;
//            break;
//        case TraceHeaderKey::SLEN:
//            val = aTraceHeaderLookup.SLEN;
//            break;
//        case TraceHeaderKey::STYP:
//            val = aTraceHeaderLookup.STYP;
//            break;
//        case TraceHeaderKey::STAS:
//            val = aTraceHeaderLookup.STAS;
//            break;
//        case TraceHeaderKey::STAE:
//            val = aTraceHeaderLookup.STAE;
//            break;
//        case TraceHeaderKey::TATYP:
//            val = aTraceHeaderLookup.TATYP;
//            break;
//        case TraceHeaderKey::AFILF:
//            val = aTraceHeaderLookup.AFILF;
//            break;
//        case TraceHeaderKey::AFILS:
//            val = aTraceHeaderLookup.AFILS;
//            break;
//        case TraceHeaderKey::NOFILF:
//            val = aTraceHeaderLookup.NOFILF;
//            break;
//        case TraceHeaderKey::NOFILS:
//            val = aTraceHeaderLookup.NOFILS;
//            break;
//        case TraceHeaderKey::LCF:
//            val = aTraceHeaderLookup.LCF;
//            break;
//        case TraceHeaderKey::HCF:
//            val = aTraceHeaderLookup.HCF;
//            break;
//        case TraceHeaderKey::LCS:
//            val = aTraceHeaderLookup.LCS;
//            break;
//        case TraceHeaderKey::HCS:
//            val = aTraceHeaderLookup.HCS;
//            break;
//        case TraceHeaderKey::YEAR:
//            val = aTraceHeaderLookup.YEAR;
//            break;
//        case TraceHeaderKey::DAY:
//            val = aTraceHeaderLookup.DAY;
//            break;
//        case TraceHeaderKey::HOUR:
//            val = aTraceHeaderLookup.HOUR;
//            break;
//        case TraceHeaderKey::MINUTE:
//            val = aTraceHeaderLookup.MINUTE;
//            break;
//        case TraceHeaderKey::SEC:
//            val = aTraceHeaderLookup.SEC;
//            break;
//        case TraceHeaderKey::TIMBAS:
//            val = aTraceHeaderLookup.TIMBAS;
//            break;
//        case TraceHeaderKey::TRWF:
//            val = aTraceHeaderLookup.TRWF;
//            break;
//        case TraceHeaderKey::GRNORS:
//            val = aTraceHeaderLookup.GRNORS;
//            break;
//        case TraceHeaderKey::GRNOFR:
//            val = aTraceHeaderLookup.GRNOFR;
//            break;
//        case TraceHeaderKey::GRNLOF:
//            val = aTraceHeaderLookup.GRNLOF;
//            break;
//        case TraceHeaderKey::GAPS:
//            val = aTraceHeaderLookup.GAPS;
//            break;
//        case TraceHeaderKey::OTRAV:
//            val = aTraceHeaderLookup.OTRAV;
//            break;
//        default:
//            throw UnsupportedFeatureException();
//    }
//    return val;
//}
