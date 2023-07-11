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
// Created by zeyad-osama on 20/09/2020.
//

#ifndef OPERATIONS_LIB_ENGINE_CONFIGURATIONS_FWI_ENGINE_CONFIGURATION_HPP
#define OPERATIONS_LIB_ENGINE_CONFIGURATIONS_FWI_ENGINE_CONFIGURATION_HPP

#include "operations/engine-configurations/interface/EngineConfigurations.hpp"

namespace operations {
    namespace configuration {

        class FWIEngineConfigurations : public EngineConfigurations {
        public:
            ~FWIEngineConfigurations() override {
                /// @todo To be implemented;
            }
        };
    } //namespace configuration
} //namespace operations

#endif //OPERATIONS_LIB_ENGINE_CONFIGURATIONS_FWI_ENGINE_CONFIGURATION_HPP
