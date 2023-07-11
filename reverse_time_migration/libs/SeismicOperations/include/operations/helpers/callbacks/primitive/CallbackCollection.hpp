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
// Created by amr-nasr on 05/11/19.
//

#ifndef OPERATIONS_LIB_CALLBACK_COLLECTION_HPP
#define OPERATIONS_LIB_CALLBACK_COLLECTION_HPP

#include <operations/helpers/callbacks/interface/Callback.hpp>

#include <vector>

namespace operations {
    namespace helpers {
        namespace callbacks {

            class CallbackCollection {
            public:
                void RegisterCallback(Callback *apCallback);

                void BeforeInitialization(common::ComputationParameters *apParameters);

                void AfterInitialization(dataunits::GridBox *apGridBox);

                void BeforeShotPreprocessing(dataunits::TracesHolder *apTraces);

                void AfterShotPreprocessing(dataunits::TracesHolder *apTraces);

                void BeforeForwardPropagation(dataunits::GridBox *apGridBox);

                void AfterForwardStep(dataunits::GridBox *apGridBox, uint time_step);

                void BeforeBackwardPropagation(dataunits::GridBox *apGridBox);

                void AfterBackwardStep(dataunits::GridBox *apGridBox, uint time_step);

                void AfterFetchStep(dataunits::GridBox *apGridBox, uint time_step);

                void BeforeShotStacking(dataunits::GridBox *apGridBox,
                                        dataunits::FrameBuffer<float> *shot_correlation);

                void AfterShotStacking(dataunits::GridBox *apGridBox,
                                       dataunits::FrameBuffer<float> *stacked_shot_correlation);

                void AfterMigration(dataunits::GridBox *apGridBox,
                                    dataunits::FrameBuffer<float> *stacked_shot_correlation);

                std::vector<Callback *> &GetCallbacks();

            private:
                std::vector<Callback *> callbacks;
            };
        } //namespace callbacks
    } //namespace operations
} //namespace operations

#endif // OPERATIONS_LIB_CALLBACK_COLLECTION_HPP
