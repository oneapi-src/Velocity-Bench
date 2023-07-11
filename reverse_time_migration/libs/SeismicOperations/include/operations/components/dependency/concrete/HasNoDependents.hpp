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
// Created by zeyad-osama on 27/09/2020.
//

#ifndef OPERATIONS_LIB_COMPONENTS_HAS_NO_DEPENDENTS_HPP
#define OPERATIONS_LIB_COMPONENTS_HAS_NO_DEPENDENTS_HPP

#include "operations/components/dependency/interface/Dependency.hpp"
#include "operations/common/ComputationParameters.hpp"

namespace operations {
    namespace components {
        namespace dependency {

            /**
             * @brief Has Dependents Component interface. All Independent Component
             * in the Seismic Framework needs to extend it in case they does not need a
             * dependent component.
             *
             * @relatedalso class HasDependents
             */
            class HasNoDependents : virtual public Dependency {
            public:
                /**
                 * @brief Destructors should be overridden to ensure correct memory management.
                 */
                ~HasNoDependents() override = default;

                /**
                 * @brief Sets the Dependent Components to use later on.
                 *
                 * @param[in] apDependentComponentsMap
                 * The designated Dependent Components to run operations on.
                 *
                 * @note Dummy implementation.
                 */
                void SetDependentComponents(
                        helpers::ComponentsMap <DependentComponent> *apDependentComponentsMap) override {};
            };
        }//namespace dependency
    }//namespace components
}//namespace operations

#endif //OPERATIONS_LIB_COMPONENTS_HAS_NO_DEPENDENTS_HPP
