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
// Created by mirna-moawad on 10/28/19.
//

#ifndef OPERATIONS_LIB_COMPONENTS_COMPUTATION_KERNELS_SECOND_ORDER_COMPUTATION_KERNEL_HPP
#define OPERATIONS_LIB_COMPONENTS_COMPUTATION_KERNELS_SECOND_ORDER_COMPUTATION_KERNEL_HPP

#include <operations/components/independents/primitive/ComputationKernel.hpp>
#include <operations/components/dependency/concrete/HasNoDependents.hpp>

namespace operations {
    namespace components {

        class SecondOrderComputationKernel : public ComputationKernel,

                                             public dependency::HasNoDependents {
        public:
            explicit SecondOrderComputationKernel(operations::configuration::ConfigurationMap *apConfigurationMap);

            SecondOrderComputationKernel(const SecondOrderComputationKernel &);

            ~SecondOrderComputationKernel() override;

            ComputationKernel *Clone() override;

            void SetComputationParameters(common::ComputationParameters *apParameters) override;

            void SetGridBox(dataunits::GridBox *apGridBox) override;

            void Step() override;

            MemoryHandler *GetMemoryHandler() override;

            void AcquireConfiguration() override;

        private:
            template<bool IS_2D_, HALF_LENGTH HALF_LENGTH_>
            void Compute();

            void InitializeVariables();

            SecondOrderComputationKernel &operator=(SecondOrderComputationKernel const &RHS) = delete;

        private:
            common::ComputationParameters *mpParameters = nullptr;

            dataunits::GridBox *mpGridBox = nullptr;

            dataunits::FrameBuffer<float> *mpCoeffX = nullptr;
            dataunits::FrameBuffer<float> *mpCoeffZ = nullptr;

            dataunits::FrameBuffer<int> *mpVerticalIdx = nullptr;

            float mCoeffXYZ;
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_COMPUTATION_KERNELS_SECOND_ORDER_COMPUTATION_KERNEL_HPP
