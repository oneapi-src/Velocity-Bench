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
// Created by zeyad-osama on 29/11/2020.
//

#ifndef OPERATIONS_LIB_CONFIGURATIONS_CONFIGURABLE_HPP
#define OPERATIONS_LIB_CONFIGURATIONS_CONFIGURABLE_HPP

#include <operations/configurations/interface/ConfigurationMap.hpp>

namespace operations {
    namespace configuration {
        /**
         * @brief Interface for any component that can be configured in the system.
         */
        class Configurable {
        public:
            /**
             * @brief Default constructor.
             */
            Configurable() = default;

            /**
             * @brief Default destructor.
             */
            virtual ~Configurable() = default;

            /**
             * @brief
             * Acquires the component configuration from a given configuration map.
             *
             * @param[in] apConfigurationMap
             * The configuration map to be used.
             */
            virtual void AcquireConfiguration() = 0;
        };
    } //namespace configuration
} //namespace operations

#endif //OPERATIONS_LIB_CONFIGURATIONS_CONFIGURABLE_HPP
