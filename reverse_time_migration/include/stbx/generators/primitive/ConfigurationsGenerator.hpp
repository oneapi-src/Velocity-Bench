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
// Created by marwan-elsafty on 21/01/2021.
//

#ifndef STBX_GENERATORS_CONFIGURATIONS_CONFIGURATIONS_GENERATOR_H
#define STBX_GENERATORS_CONFIGURATIONS_CONFIGURATIONS_GENERATOR_H

#include <operations/engine-configurations/concrete/RTMEngineConfigurations.hpp>
#include <operations/common/DataTypes.h>

#include <libraries/nlohmann/json.hpp>

#include <map>

namespace stbx {
    namespace generators {
        class ConfigurationsGenerator {
        public:
            explicit ConfigurationsGenerator(nlohmann::json &aMap);

            ~ConfigurationsGenerator() = default;

            PHYSICS
            GetPhysics();

            EQUATION_ORDER
            GetEquationOrder();

            GRID_SAMPLING
            GetGridSampling();

            APPROXIMATION
            GetApproximation();

            std::map<std::string, std::string>
            GetModelFiles();

            std::vector<std::string>
            GetTraceFiles(operations::configuration::RTMEngineConfigurations *aConfiguration);

            std::string
            GetModellingFile();

            std::string
            GetOutputFile();


        private:
            /// Map that holds configurations key value pairs
            nlohmann::json mMap;

        };
    }//namespace generators
}//namespace stbx

#endif //STBX_GENERATORS_CONFIGURATIONS_CONFIGURATIONS_GENERATOR_H

