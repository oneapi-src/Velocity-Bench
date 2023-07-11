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

#ifndef OPERATIONS_LIB_COMPONENTS_HAS_DEPENDENTS_HPP
#define OPERATIONS_LIB_COMPONENTS_HAS_DEPENDENTS_HPP

#include "operations/components/dependency/interface/Dependency.hpp"
#include "operations/components/dependency/helpers/ComponentsMap.tpp"
#include "operations/components/dependents/interface/DependentComponent.hpp"
#include "operations/common/ComputationParameters.hpp"

#include <cstdlib>

namespace operations {
    namespace components {
        namespace dependency {

            /**
             * @brief Has Dependents Component interface. All Independent Component
             * in the Seismic Framework needs to extend it in case they need a
             * dependent component.
             *
             * @relatedalso class HasNoDependents
             *
             * @note
             */
            class HasDependents : virtual public Dependency {
            public:
                /**
                 * @brief Destructors should be overridden to ensure correct memory management.
                 */
                virtual ~HasDependents() = default;

                /**
                 * @brief Sets the Dependent Components to use later on.
                 *
                 * @param[in] apDependentComponentsMap
                 * The designated Dependent Components to run operations on.
                 *
                 * @note Only components that need Dependent Components should
                 * override this function.
                 */
                virtual void SetDependentComponents(
                        operations::helpers::ComponentsMap<DependentComponent> *apDependentComponentsMap) {
                    this->mpDependentComponentsMap = apDependentComponentsMap;
                    if (this->mpDependentComponentsMap == nullptr) {
                        std::cerr << "No Dependent Components Map provided... "
                                  << "Terminating..." << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }

                /**
                 * @brief Dependent Components Map getter.
                 * @return[out] DependentComponentsMap *
                 */
                inline operations::helpers::ComponentsMap<DependentComponent> *
                GetDependentComponentsMap() const {
                    return this->mpDependentComponentsMap;
                }

            private:
                /// Dependent Components Map
                operations::helpers::ComponentsMap<DependentComponent> *mpDependentComponentsMap;
            };
        }//namespace dependency
    }//namespace components
}//namespace operations

#endif //OPERATIONS_LIB_COMPONENTS_HAS_DEPENDENTS_HPP
