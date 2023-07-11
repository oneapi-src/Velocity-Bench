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
// Created by amr-nasr on 21/10/19.
//

#ifndef OPERATIONS_LIB_COMPONENTS_SOURCE_INJECTORS_RICKER_SOURCE_INJECTOR_HPP
#define OPERATIONS_LIB_COMPONENTS_SOURCE_INJECTORS_RICKER_SOURCE_INJECTOR_HPP

#include <operations/components/independents/primitive/SourceInjector.hpp>
#include <operations/components/dependency/concrete/HasNoDependents.hpp>

namespace operations {
    namespace components {

        class RickerSourceInjector : public SourceInjector, public dependency::HasNoDependents {
        public:
            explicit RickerSourceInjector(operations::configuration::ConfigurationMap *apConfigurationMap);

            ~RickerSourceInjector() override;

            void ApplySource(uint time_step) override;

            void ApplyIsotropicField() override;

            void RevertIsotropicField() override;

            uint GetCutOffTimeStep() override;

            void SetComputationParameters(common::ComputationParameters *apParameters) override;

            void SetGridBox(dataunits::GridBox *apGridBox) override;

            void SetSourcePoint(Point3D *apSourcePoint) override;

            void AcquireConfiguration() override;

        private:
            uint GetInjectionLocation();

        private:
            common::ComputationParameters *mpParameters = nullptr;

            dataunits::GridBox *mpGridBox = nullptr;

            Point3D *mpSourcePoint = nullptr;
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_SOURCE_INJECTORS_RICKER_SOURCE_INJECTOR_HPP
