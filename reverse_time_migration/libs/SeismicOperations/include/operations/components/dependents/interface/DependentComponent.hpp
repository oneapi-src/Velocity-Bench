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

#ifndef OPERATIONS_LIB_COMPONENTS_DEPENDENT_COMPONENT_HPP
#define OPERATIONS_LIB_COMPONENTS_DEPENDENT_COMPONENT_HPP

#include "operations/configurations/interface/Configurable.hpp"
#include "operations/common/ComputationParameters.hpp"

namespace operations {
    namespace components {
        /**
         * @brief Dependent Component interface. All Dependent Component
         * in the Seismic Framework needs to extend it.
         *
         * @note Each engine comes with it's own Timer and Logger. Logger
         * Channel should be initialized at each concrete implementation
         * and should be destructed at each destructor.
         */
        class DependentComponent : public configuration::Configurable {
        public:
            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            virtual ~DependentComponent() = default;

            /**
             * @brief Sets the computation parameters to be used for the component.
             *
             * @param[in] apParameters
             * Parameters of the simulation independent from the grid.
             */
            virtual void SetComputationParameters(common::ComputationParameters *apParameters) = 0;

        protected:
            /// Configurations Map
            operations::configuration::ConfigurationMap *mpConfigurationMap;
        };
    }//namespace components
}//namespace operations

#endif //OPERATIONS_LIB_COMPONENTS_DEPENDENT_COMPONENT_HPP
