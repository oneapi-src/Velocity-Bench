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

#ifndef OPERATIONS_LIB_COMPONENTS_DEPENDENCY_HPP
#define OPERATIONS_LIB_COMPONENTS_DEPENDENCY_HPP

#include "operations/components/dependency/helpers/ComponentsMap.tpp"

namespace operations {
    namespace components {
        namespace dependency {

            class Dependency {
            public:
                /**
                 * @brief Destructors should be overridden to ensure correct memory management.
                 */
                virtual ~Dependency() = default;;

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
                        operations::helpers::ComponentsMap<DependentComponent> *apDependentComponentsMap) = 0;
            };
        }//namespace dependency
    }//namespace components
}//namespace operations

#endif //OPERATIONS_LIB_COMPONENTS_DEPENDENCY_HPP
