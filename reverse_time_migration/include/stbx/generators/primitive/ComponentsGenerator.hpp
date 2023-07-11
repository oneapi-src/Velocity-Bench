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
// Created by marwan-elsafty on 18/01/2021.
//

#ifndef STBX_GENERATORS_COMPONENTS_COMPONENTS_GENERATOR
#define STBX_GENERATORS_COMPONENTS_COMPONENTS_GENERATOR

#include <operations/configurations/concrete/JSONConfigurationMap.hpp>
#include <operations/common/DataTypes.h>
#include <operations/components/Components.hpp>

namespace stbx {
    namespace generators {

        class ComponentsGenerator {
        public:
            explicit ComponentsGenerator(const nlohmann::json &aMap,
                                         EQUATION_ORDER aOrder,
                                         GRID_SAMPLING aSampling,
                                         APPROXIMATION aApproximation);

            ~ComponentsGenerator() = default;

            operations::components::ComputationKernel *
            GenerateComputationKernel();

            operations::components::ModelHandler *
            GenerateModelHandler();

            operations::components::SourceInjector *
            GenerateSourceInjector();

            operations::components::BoundaryManager *
            GenerateBoundaryManager();

            operations::components::ForwardCollector *
            GenerateForwardCollector(const std::string &write_path);

            operations::components::MigrationAccommodator *
            GenerateMigrationAccommodator();

            operations::components::TraceManager *
            GenerateTraceManager();

            void GenerateModellingConfigurationParser();
            operations::components::ModellingConfigurationParser *GetModellingConfigurationParser() { return m_ModellingConfigurationParser; } 


            operations::components::TraceWriter *
            GenerateTraceWriter();


        private:
            operations::configuration::JSONConfigurationMap *
            TruncateMap(const std::string &aComponentName);

            nlohmann::json
            GetWaveMap();

            void CheckFirstOrder();

        private:
            /// Map that holds configurations key value pairs
            nlohmann::json mMap;
            EQUATION_ORDER mOrder;
            GRID_SAMPLING mSampling;
            APPROXIMATION mApproximation;
            operations::components::ModellingConfigurationParser *m_ModellingConfigurationParser;
        };
    }// namespace generators
}//namesapce stbx


#endif //STBX_GENERATORS_COMPONENTS_COMPONENTS_GENERATOR
