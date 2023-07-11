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
// Created by marwan-elsafty on 22/12/2020.
//

#ifndef OPERATIONS_LIB_CONFIGURATIONS_MAP_KEYS_HPP
#define OPERATIONS_LIB_CONFIGURATIONS_MAP_KEYS_HPP

namespace operations {
    namespace configuration {

#define OP_K_PROPRIETIES               "properties"
#define OP_K_USE_TOP_LAYER             "use-top-layer"
#define OP_K_SHOT_STRIDE               "shot-stride"
#define OP_K_REFLECT_COEFFICIENT       "reflect-coeff"
#define OP_K_SHIFT_RATIO               "shift-ratio"
#define OP_K_RELAX_COEFFICIENT         "relax-coeff"
#define OP_K_ZFP_TOLERANCE             "zfp-tolerance"
#define OP_K_ZFP_PARALLEL              "zfp-parallel"
#define OP_K_ZFP_RELATIVE              "zfp-relative"
#define OP_K_WRITE_PATH                "write-path"
#define OP_K_COMPRESSION               "compression"
#define OP_K_COMPRESSION_TYPE          "compression-type"
#define OP_K_BOUNDARY_SAVING           "boundary-saving"
#define OP_K_COMPENSATION              "compensation"
#define OP_K_COMPENSATION_NONE         "none"
#define OP_K_COMPENSATION_COMBINED     "combined"
#define OP_K_COMPENSATION_RECEIVER     "receiver"
#define OP_K_COMPENSATION_SOURCE       "source"
#define OP_K_COMPENSATION              "compensation"
#define OP_K_TYPE                      "type"
#define OP_K_INTERPOLATION             "interpolation"
#define OP_K_NONE                      "none"
#define OP_K_SPLINE                    "spline"

    } //namespace configuration
} //namespace operations

#endif //OPERATIONS_LIB_CONFIGURATIONS_MAP_KEYS_HPP
