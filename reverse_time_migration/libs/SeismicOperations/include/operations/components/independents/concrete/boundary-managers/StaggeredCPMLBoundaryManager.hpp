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
// Created by amr-nasr on 21/10/2019.
//

#ifndef OPERATIONS_LIB_COMPONENTS_BOUNDARY_MANAGERS_STAGGERED_CPML_BOUNDARY_MANAGER_HPP
#define OPERATIONS_LIB_COMPONENTS_BOUNDARY_MANAGERS_STAGGERED_CPML_BOUNDARY_MANAGER_HPP

#include <operations/components/independents/concrete/boundary-managers/extensions/Extension.hpp>
#include <operations/components/independents/primitive/BoundaryManager.hpp>
#include <operations/components/dependency/concrete/HasNoDependents.hpp>

#include <memory-manager/MemoryManager.h>

#include <math.h>
#include <vector>

namespace operations {
    namespace components {

        class StaggeredCPMLBoundaryManager : public BoundaryManager,
                                             public dependency::HasNoDependents {
        public:
            explicit StaggeredCPMLBoundaryManager(operations::configuration::ConfigurationMap *apConfigurationMap);

            ~StaggeredCPMLBoundaryManager() override;

            void ApplyBoundary(uint kernel_id) override;

            void ExtendModel() override;

            void ReExtendModel() override;

            void SetComputationParameters(common::ComputationParameters *apParameters) override;

            void SetGridBox(dataunits::GridBox *apGridBox) override;

            void AdjustModelForBackward() override;

            void AcquireConfiguration() override;

        private:
            void InitializeExtensions();

            void FillCPMLCoefficients(float *coeff_a, float *coeff_b, int boundary_length,
                                      float dh, float dt, float max_vel, float shift_ratio,
                                      float reflect_coeff, float relax_cp);

            void ZeroAuxiliaryVariables();

        private:
            common::ComputationParameters *mpParameters = nullptr;

            dataunits::GridBox *mpGridBox = nullptr;

            std::vector<addons::Extension *> mvExtensions;

            bool mUseTopLayer;

            float mMaxVelocity;

            float mReflectCoefficient = 0.1;
            float mShiftRatio = 0.1;
            float mRelaxCoefficient = 1;

            dataunits::FrameBuffer<float> *small_a_x = nullptr;
            dataunits::FrameBuffer<float> *small_a_z = nullptr;
            dataunits::FrameBuffer<float> *small_b_x = nullptr;
            dataunits::FrameBuffer<float> *small_b_z = nullptr;

            dataunits::FrameBuffer<float> *auxiliary_vel_x_left = nullptr;
            dataunits::FrameBuffer<float> *auxiliary_vel_x_right = nullptr;
            dataunits::FrameBuffer<float> *auxiliary_vel_z_up = nullptr;
            dataunits::FrameBuffer<float> *auxiliary_vel_z_down = nullptr;

            dataunits::FrameBuffer<float> *auxiliary_ptr_x_left = nullptr;
            dataunits::FrameBuffer<float> *auxiliary_ptr_x_right = nullptr;

            dataunits::FrameBuffer<float> *auxiliary_ptr_z_up = nullptr;
            dataunits::FrameBuffer<float> *auxiliary_ptr_z_down = nullptr;
        };
    }//namespace components
}//namespace operations

#endif //OPERATIONS_LIB_COMPONENTS_BOUNDARY_MANAGERS_STAGGERED_CPML_BOUNDARY_MANAGER_HPP
