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
// Created by amr-nasr on 18/11/2019.
//

#ifndef OPERATIONS_LIB_COMPONENTS_BOUNDARY_MANAGERS_CPML_BOUNDARY_MANAGER_HPP
#define OPERATIONS_LIB_COMPONENTS_BOUNDARY_MANAGERS_CPML_BOUNDARY_MANAGER_HPP

#include <operations/components/independents/concrete/boundary-managers/extensions/Extension.hpp>
#include <operations/components/independents/primitive/BoundaryManager.hpp>
#include <operations/components/dependency/concrete/HasNoDependents.hpp>

#include <memory-manager/MemoryManager.h>

#include <cmath>

namespace operations {
    namespace components {

        class CPMLBoundaryManager : public BoundaryManager,
                                    public dependency::HasNoDependents {
        public:
            explicit CPMLBoundaryManager(
                    operations::configuration::ConfigurationMap *apConfigurationMap);

            ~CPMLBoundaryManager() override;

            void ApplyBoundary(uint kernel_id) override;

            void ExtendModel() override;

            void ReExtendModel() override;

            void AdjustModelForBackward() override;

            void SetComputationParameters(common::ComputationParameters *apParameters) override;

            void SetGridBox(dataunits::GridBox *apGridBox) override;

            void AcquireConfiguration() override;

        private:
            template<int DIRECTION_>
            void FillCPMLCoefficients();

            template<int DIRECTION_, bool OPPOSITE_, int HALF_LENGTH_>
            void CalculateFirstAuxiliary();

            template<int DIRECTION_, bool OPPOSITE_, int HALF_LENGTH_>
            void CalculateCPMLValue();

            template<int HALF_LENGTH_>
            void ApplyAllCPML();

            void InitializeVariables();

            void ResetVariables();

            CPMLBoundaryManager           (CPMLBoundaryManager const &RHS) = delete;
            CPMLBoundaryManager &operator=(CPMLBoundaryManager const &RHS) = delete;

        private:
            common::ComputationParameters *mpParameters = nullptr;

            dataunits::GridBox *mpGridBox = nullptr;

            addons::Extension *mpExtension = nullptr;

            dataunits::FrameBuffer<float> *coeff_a_x;
            dataunits::FrameBuffer<float> *coeff_b_x;
            dataunits::FrameBuffer<float> *coeff_a_z;
            dataunits::FrameBuffer<float> *coeff_b_z;

            dataunits::FrameBuffer<float> *aux_1_x_up;
            dataunits::FrameBuffer<float> *aux_1_x_down;
            dataunits::FrameBuffer<float> *aux_1_z_up;
            dataunits::FrameBuffer<float> *aux_1_z_down;

            dataunits::FrameBuffer<float> *aux_2_x_up;
            dataunits::FrameBuffer<float> *aux_2_x_down;
            dataunits::FrameBuffer<float> *aux_2_z_up;
            dataunits::FrameBuffer<float> *aux_2_z_down;

            float max_vel;

            float mRelaxCoefficient;
            float mShiftRatio;
            float mReflectCoefficient;
            bool mUseTopLayer;
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_BOUNDARY_MANAGERS_CPML_BOUNDARY_MANAGER_HPP
