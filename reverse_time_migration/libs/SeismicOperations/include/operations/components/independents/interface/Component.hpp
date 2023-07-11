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
// Created by amr-nasr on 09/12/2019.
//

#ifndef OPERATIONS_LIB_COMPONENTS_COMPONENT_HPP
#define OPERATIONS_LIB_COMPONENTS_COMPONENT_HPP

#include <operations/configurations/interface/Configurable.hpp>
#include <operations/components/dependency/interface/Dependency.hpp>
#include <operations/components/dependency/helpers/ComponentsMap.tpp>
#include <operations/components/dependents/interface/DependentComponent.hpp>
#include <operations/common/ComputationParameters.hpp>
#include <operations/data-units/concrete/holders/GridBox.hpp>

#include <timer/Timer.h>

namespace operations {
    namespace components {

        /**
         * @brief Component interface. All components in the Seismic Framework
         * needs to extend it.
         *
         * @note Each engine comes with it's own Timer and Logger. Logger
         * Channel should be initialized at each concrete implementation
         * and should be destructed at each destructor.
         */
        class Component : virtual public dependency::Dependency,
                          public configuration::Configurable {
        public:
            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            virtual ~Component() = default;

            /**
             * @brief Sets the computation parameters to be used for the component.
             *
             * @param[in] apParameters
             * Parameters of the simulation independent from the grid.
             */
            virtual void SetComputationParameters(common::ComputationParameters *apParameters) = 0;

            /**
             * @brief Sets the grid box to operate on.
             *
             * @param[in] apGridBox
             * The designated grid box to run operations on.
             */
            virtual void SetGridBox(dataunits::GridBox *apGridBox) = 0;

            /**
             * @brief Sets Components Map. Let al components be aware of
             * each other.
             *
             * @param[in] apComponentsMap
             */
            void SetComponentsMap(operations::helpers::ComponentsMap<Component> *apComponentsMap) {
                this->mpComponentsMap = apComponentsMap;
            }

            virtual Component *Clone() = 0; 

        protected:
            /// Timer
            Timer *mpTimer;
            /// Dependent Components Map
            operations::helpers::ComponentsMap<DependentComponent> *mpDependentComponentsMap;
            /// Independent Components Map
            operations::helpers::ComponentsMap<Component> *mpComponentsMap;
            /// Configurations Map
            operations::configuration::ConfigurationMap *mpConfigurationMap;
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_COMPONENT_HPP
