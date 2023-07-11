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
// Created by amr-nasr on 02/11/2019.
//

#ifndef OPERATIONS_LIB_CALLBACK_HPP
#define OPERATIONS_LIB_CALLBACK_HPP

#include "operations/common/DataTypes.h"
#include "operations/common/ComputationParameters.hpp"
#include "operations/data-units/concrete/holders/GridBox.hpp"
#include "operations/data-units/concrete/holders/TracesHolder.hpp"

namespace operations {
    namespace helpers {
        namespace callbacks {

            class Callback {
            public:
                virtual void BeforeInitialization(common::ComputationParameters *apParameters) = 0;

                virtual void AfterInitialization(dataunits::GridBox *apGridBox) = 0;

                virtual void BeforeShotPreprocessing(dataunits::TracesHolder *apTraces) = 0;

                virtual void AfterShotPreprocessing(dataunits::TracesHolder *apTraces) = 0;

                virtual void BeforeForwardPropagation(dataunits::GridBox *apGridBox) = 0;

                virtual void AfterForwardStep(dataunits::GridBox *apGridBox, uint time_step) = 0;

                virtual void BeforeBackwardPropagation(dataunits::GridBox *apGridBox) = 0;

                virtual void AfterBackwardStep(dataunits::GridBox *apGridBox, uint time_step) = 0;

                virtual void AfterFetchStep(dataunits::GridBox *apGridBox, uint time_step) = 0;

                virtual void
                BeforeShotStacking(dataunits::GridBox *apGridBox, dataunits::FrameBuffer<float> *shot_correlation) = 0;

                virtual void AfterShotStacking(dataunits::GridBox *apGridBox,
                                               dataunits::FrameBuffer<float> *stacked_shot_correlation) = 0;

                virtual void AfterMigration(dataunits::GridBox *apGridBox,
                                            dataunits::FrameBuffer<float> *stacked_shot_correlation) = 0;
            };
        } //namespace callbacks
    } //namespace operations
} //namespace operations

#endif // OPERATIONS_LIB_CALLBACK_HPP
