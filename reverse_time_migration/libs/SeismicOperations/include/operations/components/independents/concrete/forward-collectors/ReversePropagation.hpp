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
// Created by amr-nasr on 1/20/20.
//

#ifndef OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_REVERSE_PROPAGATION_HPP
#define OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_REVERSE_PROPAGATION_HPP

#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>
#include <operations/components/independents/primitive/ForwardCollector.hpp>
#include <operations/components/dependency/concrete/HasDependents.hpp>

#include <operations/components/independents/primitive/ComputationKernel.hpp>
#include <memory-manager/MemoryManager.h>

#include <cstdlib>
#include <cstring>

namespace operations {
    namespace components {

        class ReversePropagation : public ForwardCollector,
                                   public dependency::HasDependents {
        public:
            explicit ReversePropagation(operations::configuration::ConfigurationMap *apConfigurationMap);

            ~ReversePropagation() override;

            void SetComputationParameters(common::ComputationParameters *apParameters) override;

            void SetGridBox(dataunits::GridBox *apGridBox) override;

            void SetDependentComponents(
                    operations::helpers::ComponentsMap<DependentComponent> *apDependentComponentsMap) override;

            void FetchForward() override;

            void SaveForward() override;

            void ResetGrid(bool is_forward_run) override;

            dataunits::GridBox *GetForwardGrid() override;

            void AcquireConfiguration() override;

        private:
            /**
             * @brief Used by reverse injection propagation mode.
             */
            void Inject();

        private:
            common::ComputationParameters *mpParameters = nullptr;

            ComputationKernel *mpComputationKernel = nullptr;

            WaveFieldsMemoryHandler *mpWaveFieldsMemoryHandler = nullptr;

            /*
             * Propagation Properties.
             */

            dataunits::GridBox *mpMainGridBox = nullptr;

            dataunits::GridBox *mpInternalGridBox = nullptr;

            /*
             * Injection Properties.
             */

            bool mInjectionEnabled;

            float *mpBackupBoundaries = nullptr;

            uint mTimeStep;

            uint mBoundariesSize;

            ReversePropagation           (ReversePropagation const &RHS) = delete;
            ReversePropagation &operator=(ReversePropagation const &RHS) = delete;
        };
    }//namespace components
}//namespace operations

#endif //OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_REVERSE_PROPAGATION_HPP
