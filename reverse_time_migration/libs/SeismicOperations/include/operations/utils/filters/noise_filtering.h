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
// Created by amr-nasr on 24/06/2020.
//

#ifndef OPERATIONS_LIB_UTILS_NOISE_FILTERING_H
#define OPERATIONS_LIB_UTILS_NOISE_FILTERING_H

namespace operations {
    namespace utils {
        namespace filters {

            typedef unsigned int uint;

            void filter_stacked_correlation(float *input_frame, float *output_frame,
                                            uint nx, uint ny, uint nz,
                                            float dx, float dz, float dy);

            void apply_laplace_filter(float *input_frame, float *output_frame,
                                      unsigned int nx, unsigned int ny, unsigned int nz);
        } //namespace filters
    } //namespace utils
} //namespace operations

#endif //OPERATIONS_LIB_UTILS_NOISE_FILTERING_H
