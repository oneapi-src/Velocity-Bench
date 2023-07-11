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
// Created by ahmed-ayyad on 17/01/2021.
//

#ifndef OPERATIONS_LIB_UTILS_UTILS_SAMPLING_SAMPLER_HPP
#define OPERATIONS_LIB_UTILS_UTILS_SAMPLING_SAMPLER_HPP

#include <operations/common/ComputationParameters.hpp>
#include <operations/data-units/concrete/holders/GridBox.hpp>
#include <operations/data-units/concrete/holders/FrameBuffer.hpp>

using namespace operations::dataunits;
using namespace operations::common;

namespace operations {
    namespace utils {
        namespace sampling {

            class Sampler {
            public:
                static void
                Resize(
                        float *input, float *output,
                        GridSize *apInputGridBox, GridSize *apOutputGridBox,
                        ComputationParameters *apParameters);

                static void
                CalculateAdaptiveCellDimensions(GridBox *apGridBox,
                                                ComputationParameters *apParameters,
                                                int minimum_velocity);
            };

        } //namespace sampling
    } //namespace utils
} //namespace operations


#endif /* OPERATIONS_LIB_UTILS_UTILS_SAMPLING_SAMPLER_HPP */
