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
// Created by mirna-moawad on 1/15/20.
//

#ifndef OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_TWO_PROPAGATION_HPP
#define OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_TWO_PROPAGATION_HPP

#include <operations/components/independents/concrete/forward-collectors/file-handler/file_handler.h>
#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>
#include <operations/components/independents/primitive/ForwardCollector.hpp>
#include <operations/components/dependency/concrete/HasDependents.hpp>

#include <memory-manager/MemoryManager.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace operations {
    namespace components {

        class TwoPropagation : public ForwardCollector,
                               public dependency::HasDependents {
        public:
            explicit TwoPropagation(operations::configuration::ConfigurationMap *apConfigurationMap);

            ~TwoPropagation() override;

            void SetComputationParameters(common::ComputationParameters *apParameters) override;

            void SetGridBox(dataunits::GridBox *apGridBox) override;

            void SetDependentComponents(
                    operations::helpers::ComponentsMap<DependentComponent> *apDependentComponentsMap) override;

            void FetchForward() override;

            void SaveForward() override;

            void ResetGrid(bool aIsForwardRun) override;

            dataunits::GridBox *GetForwardGrid() override;

            void AcquireConfiguration() override;

        private:
            common::ComputationParameters *mpParameters = nullptr;

            dataunits::GridBox *mpMainGridBox = nullptr;

            dataunits::GridBox *mpInternalGridBox = nullptr;

            WaveFieldsMemoryHandler *mpWaveFieldsMemoryHandler = nullptr;

            dataunits::FrameBuffer<float> *mpForwardPressure = nullptr;

            float *mpForwardPressureHostMemory = nullptr;

            float *mpTempPrev = nullptr;

            float *mpTempCurr = nullptr;

            float *mpTempNext = nullptr;

            bool mIsMemoryFit;

            uint mTimeCounter;

            unsigned long long mMaxNT;

            unsigned long long mMaxDeviceNT;

            unsigned int mpMaxNTRatio;

            std::string mWritePath;

            bool mIsCompression;

            /* ZFP Properties. */

            int mZFP_Parallel;

            bool mZFP_IsRelative;

            float mZFP_Tolerance;

            TwoPropagation (TwoPropagation const &RHS) = delete;
            TwoPropagation &operator=(TwoPropagation const &RHS) = delete;
        };
    }//namespace components
}//namespace operations

#endif //OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_TWO_PROPAGATION_HPP
