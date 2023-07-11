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

#include  <stbx/generators/primitive/ComponentsGenerator.hpp>

#include <stbx/test-utils/utils.h>

#include <libraries/catch/catch.hpp>
#include <libraries/nlohmann/json.hpp>

#include <string>

using namespace stbx::generators;
using namespace stbx::testutils;

using namespace operations::common;
using namespace operations::configuration;
using namespace operations::components;
using namespace operations::exceptions;


void TEST_CASE_COMPONENTS_GENERATOR() {
    nlohmann::json map = R"(
{
   "boundary-manager": {
      "type": "none",
      "properties": {
        "use-top-layer": false
      }
    },
    "migration-accommodator": {
      "type": "cross-correlation",
      "properties": {
        "compensation": "no"
      }
    },
    "forward-collector": {
      "type": "three"
    },
    "trace-manager": {
      "type": "segy",
      "properties": {
        "shot-stride" : 2
      }
    },
    "source-injector": {
      "type": "ricker"
    },
    "model-handler": {
      "type": "segy"
    },
    "trace-writer": {
          "type": "binary"
    },
    "modelling-configuration-parser": {
      "type": "text"
    }

}
)"_json;
    APPROXIMATION approximation = ISOTROPIC;
    EQUATION_ORDER order = SECOND;
    GRID_SAMPLING sampling = UNIFORM;

    auto components_generator = new ComponentsGenerator(map, order, sampling, approximation);

    SECTION("GenerateComputationKernel") {
        auto computation_kernel = components_generator->GenerateComputationKernel();
        REQUIRE(instanceof<ComputationKernel>(computation_kernel));
        delete computation_kernel;
    }

    SECTION("GenerateModelHandler") {
        auto model_handler = components_generator->GenerateModelHandler();
        REQUIRE(instanceof<ModelHandler>(model_handler));
        delete model_handler;
    }

    SECTION("GenerateSourceInjector") {
        auto source_injector = components_generator->GenerateSourceInjector();
        REQUIRE(instanceof<SourceInjector>(source_injector));
        delete source_injector;
    }

    SECTION("GenerateBoundaryManager") {
        auto boundary_manager = components_generator->GenerateBoundaryManager();
        REQUIRE(instanceof<BoundaryManager>(boundary_manager));
        delete boundary_manager;
    }

    SECTION("GenerateForwardCollector") {
        auto forward_collector = components_generator->GenerateForwardCollector(WRITE_PATH);
        REQUIRE(instanceof<ForwardCollector>(forward_collector));
    }

    SECTION("GenerateMigrationAccommodator") {
        auto migration_accommodator = components_generator->GenerateMigrationAccommodator();
        REQUIRE(instanceof<MigrationAccommodator>(migration_accommodator));
    }

    SECTION("GenerateTraceManager") {
        auto trace_manager = components_generator->GenerateTraceManager();
        REQUIRE(instanceof<TraceManager>(trace_manager));
        delete trace_manager;
    }

    SECTION("GenerateModellingConfigurationParser") {
        auto modelling_configuration_parser =
                components_generator->GenerateModellingConfigurationParser();
        REQUIRE(instanceof<ModellingConfigurationParser>(modelling_configuration_parser));
        delete modelling_configuration_parser;
    }

    SECTION("GenerateTraceWriter") {
        auto trace_writer = components_generator->GenerateTraceWriter();
        REQUIRE(instanceof<TraceWriter>(trace_writer));
        delete trace_writer;
    }
}

TEST_CASE("ComponentsGenerator Class", "[Generator],[ComponentsGenerator]") {
    TEST_CASE_COMPONENTS_GENERATOR();
}
