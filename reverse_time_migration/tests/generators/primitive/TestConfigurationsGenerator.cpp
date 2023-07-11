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
// Created by marwan-elsafty on 24/01/2021.
//

#include <stbx/generators/primitive/ConfigurationsGenerator.hpp>

#include <libraries/catch/catch.hpp>
#include <libraries/nlohmann/json.hpp>

using namespace stbx::generators;
using namespace operations::configuration;
using namespace std;


void TEST_CASE_CONFIGURATIONS_GENERATOR() {
    SECTION("WaveGetter") {
        nlohmann::json map = R"(
           {
  "wave": {
    "physics": "acoustic",
    "approximation": "isotropic",
    "equation-order": "first",
    "grid-sampling": "uniform"
  }
}
    )"_json;
        auto *configurations_generator = new ConfigurationsGenerator(map);

        SECTION("GetPhysics Function Testing") {
            PHYSICS physics = configurations_generator->GetPhysics();
            REQUIRE(physics == ACOUSTIC);
        }

        SECTION("GetEquationOrder Function Testing") {
            EQUATION_ORDER order = configurations_generator->GetEquationOrder();
            REQUIRE(order == FIRST);
        }

        SECTION("GetGridSampling Function Testing") {
            GRID_SAMPLING sampling = configurations_generator->GetGridSampling();
            REQUIRE(sampling == UNIFORM);
        }

        SECTION("GetApproximation") {
            APPROXIMATION approximation = configurations_generator->GetApproximation();
            REQUIRE(approximation == ISOTROPIC);
        }
    }

    SECTION("FileGetter Function Testing") {
        nlohmann::json map = R"(
         {
  "traces": {
    "min": 0,
    "max": 601,
    "sort-type": "CSR",
    "paths": [
      "path test"
    ]
  },
  "models": {
    "velocity": "velocity test",
    "density": "density test",
    "delta": "delta test",
    "epsilon": "epsilon test",
    "theta": "theta test",
    "phi": "phi test"
  },
  "modelling-file": "modelling-file test",
  "output-file": "output-file test",

  "wave": {
    "physics": "acoustic",
    "approximation": "isotropic",
    "equation-order": "first",
    "grid-sampling": "uniform"
  }
}
    )"_json;
        auto *configurations_generator = new ConfigurationsGenerator(map);

        SECTION("GetModelFiles Function Testing") {
            std::map<string, string> model_files = configurations_generator->GetModelFiles();
            REQUIRE(model_files["velocity"] == "velocity test");
            REQUIRE(model_files["density"] == "density test");
            REQUIRE(model_files["delta"] == "delta test");
            REQUIRE(model_files["theta"] == "theta test");
            REQUIRE(model_files["phi"] == "phi test");
        }

        SECTION("GetTraceFiles Function Testing") {
            auto *configuration = new RTMEngineConfigurations();
            vector<string> traces_files = configurations_generator->GetTraceFiles(configuration);

            REQUIRE(configuration->GetSortMin() == 0);
            REQUIRE(configuration->GetSortMax() == 601);
            REQUIRE(configuration->GetSortKey() == "CSR");
            REQUIRE(traces_files.back() == "path test");
        }

        SECTION("GetModellingFile Function Testing") {
            string file_name = configurations_generator->GetModellingFile();
            REQUIRE(file_name == "modelling-file test");
        }

        SECTION("GetOutputFile Function Testing") {
            string filename = configurations_generator->GetOutputFile();
            REQUIRE(filename == "output-file test");
        }
    }
}

TEST_CASE("ConfigurationsGenerator", "[Generator],[ConfigurationsGenerator]") {
    TEST_CASE_CONFIGURATIONS_GENERATOR();
}
