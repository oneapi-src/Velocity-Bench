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
// Created by mirna-moawad on 1/9/20.
//

#ifndef OPERATIONS_LIB_COMPONENTS_COMPUTATION_KERNELS_STAGGERED_ORDER_COMPUTATION_KERNEL_HPP
#define OPERATIONS_LIB_COMPONENTS_COMPUTATION_KERNELS_STAGGERED_ORDER_COMPUTATION_KERNEL_HPP

#include <operations/components/independents/primitive/ComputationKernel.hpp>
#include <operations/components/dependency/concrete/HasNoDependents.hpp>

namespace operations {
    namespace components {

        class StaggeredComputationKernel : public ComputationKernel, public dependency::HasNoDependents {
        public:
            explicit StaggeredComputationKernel(operations::configuration::ConfigurationMap *apConfigurationMap);

            StaggeredComputationKernel(const StaggeredComputationKernel &aStaggeredComputationKernel);

            ~StaggeredComputationKernel() override;

            ComputationKernel *Clone() override;

            void SetComputationParameters(common::ComputationParameters *apParameters) override;

            void SetGridBox(dataunits::GridBox *apGridBox) override;

            void Step() override;

            MemoryHandler *GetMemoryHandler() override;

            void AcquireConfiguration() override;

        private:
            template<bool IS_FORWARD_, HALF_LENGTH HALF_LENGTH_>
            void Compute();

        private:
            common::ComputationParameters *mpParameters = nullptr;

            dataunits::GridBox *mpGridBox = nullptr;
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_COMPUTATION_KERNELS_STAGGERED_ORDER_COMPUTATION_KERNEL_HPP
