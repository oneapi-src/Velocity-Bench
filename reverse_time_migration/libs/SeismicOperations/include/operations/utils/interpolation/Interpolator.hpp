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
// Created by zeyad-osama on 08/07/2020.
//

#ifndef OPERATIONS_LIB_UTILS_INTERPOLATION_H
#define OPERATIONS_LIB_UTILS_INTERPOLATION_H

#include <operations/data-units/concrete/holders/TracesHolder.hpp>

namespace operations {
    namespace utils {
        namespace interpolation {

            class Interpolator {
            public:
                /**
                 * @brief Takes a Trace struct and modifies it's attributes to fit with the new
                 * actual_nt
                 * @param apTraceHolder            Traces struct
                 * @param actual_nt                To be interpolated
                 * @param total_time               Total time of forward propagation
                 * @param aInterpolation           Interpolation type
                 * @return Interpolated apTraceHolder in Traces struct [Not needed]
                 */
                static float *
                Interpolate(dataunits::TracesHolder *apTraceHolder, uint actual_nt, float total_time,
                            INTERPOLATION aInterpolation = NONE);

                static float *
                InterpolateLinear(dataunits::TracesHolder *apTraceHolder, uint actual_nt, float total_time);

                static void
                InterpolateTrilinear(float *old_grid, float *new_grid,
                                     int old_nx, int old_nz, int old_ny,
                                     int new_nx, int new_nz, int new_ny,
                                     int bound_length,
                                     int half_length);
            };

        } //namespace interpolation
    } //namespace utils
} //namespace operations

#endif // OPERATIONS_LIB_UTILS_INTERPOLATION_H
