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
// Created by marwan-elsafty on 17/01/2021.
//

#include <stbx/generators/Generator.hpp>

#include <stbx/generators/primitive/CallbacksGenerator.hpp>
#include <stbx/generators/concrete/computation-parameters/computation_parameters_generator.h>
#include <stbx/generators/primitive/ConfigurationsGenerator.hpp>
#include <stbx/test-utils/utils.h>

#include <libraries/catch/catch.hpp>

using namespace stbx::agents;
using namespace stbx::writers;
using namespace stbx::generators;
using namespace stbx::testutils;

using namespace operations::common;
using namespace operations::helpers::callbacks;
using namespace operations::configuration;
using namespace operations::components;
using namespace operations::exceptions;

using namespace std;


void TEST_CASE_GENERATOR() {
    nlohmann::json ground_truth_map = R"(
           {
  "callbacks": {
    "su": {
      "enable": true,
      "show-each": 200,
      "little-endian": false
    },
    "csv": {
      "enable": true,
      "show-each": 200
    },
    "image": {
      "enable": true,
      "show-each": 200,
      "percentile": 98.5
    },
    "norm": {
      "enable": true,
      "show-each": 200
    },
    "bin": {
      "enable": true,
      "show-each": 200
    },
    "segy": {
      "enable": true,
      "show-each": 200
    },
    "writers": {
      "migration": {
        "enable": true
      },
      "parameters": {
        "enable": true,
        "list": ["velocity", "density"]
      },
      "traces-raw": {
        "enable": true
      },
      "traces-preprocessed": {
        "enable": true
      },
      "re-extended-parameters": {
        "enable": true,
        "list": ["velocity", "density"]
      },
      "each-stacked-shot": {
        "enable": true
      },
      "single-shot-correlation": {
        "enable": true
      },
      "backward": {
        "enable": true
      },
      "forward": {
        "enable": true
      },
      "reverse": {
        "enable": true
      }
    }
  },
  "computation-parameters": {
    "stencil-order": 8,
    "boundary-length": 20,
    "source-frequency": 20,
    "isotropic-radius": 5,
    "dt-relax": 0.9,
    "algorithm": "cpu",
    "Device": "none",
    "cache-blocking": {
      "block-x": 128,
      "block-z": 16,
      "block-y": 1,
      "cor-block": 256
    },
    "window": {
      "enable": true,
      "left": 250,
      "right": 250,
      "depth": 500,
      "front": 0,
      "back": 0
    }
  },
  "traces": {
    "min": 601,
    "max": 601,
    "sort-type": "CSR",
    "paths": [
      "data/iso/shots/shots0601_0800.segy"
    ]
  },
  "models": {
    "velocity": "data/iso/params/vel_z6.25m_x12.5m_exact.segy"
  },
  "wave": {
    "physics": "acoustic",
    "approximation": "isotropic",
    "equation-order": "second",
    "grid-sampling": "uniform"
  },
  "components": {
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
  },
  "interpolation": {
    "type": "none"
  },

  "pipeline": {
    "agent": {
      "type": "normal"
    },
    "writer": {
      "type": "normal"
    }
  },
  "modelling-file": "workloads/synthetic_model/modelling.json",
  "output-file": "data/shot.trace"

}
    )"_json;

    auto *generator = new Generator(ground_truth_map);

    SECTION("Instance not null") {
        REQUIRE(instanceof<Generator>(generator));
        REQUIRE(generator != nullptr);
    }

    SECTION("GenerateCallbacks Function Testing") {
        REQUIRE(instanceof<CallbackCollection>(generator->GenerateCallbacks(WRITE_PATH)));
    }

    SECTION("GenerateModellingEngineConfiguration Function Testing") {
        auto *configuration = generator->GenerateModellingEngineConfiguration(WRITE_PATH);

        REQUIRE(instanceof<ModellingEngineConfigurations>(configuration));
        REQUIRE(instanceof<ComputationKernel>(configuration->GetComputationKernel()));
        REQUIRE(instanceof<ModelHandler>(configuration->GetModelHandler()));
        REQUIRE(instanceof<SourceInjector>(configuration->GetSourceInjector()));
        REQUIRE(instanceof<BoundaryManager>(configuration->GetBoundaryManager()));
        REQUIRE(instanceof<ModellingConfigurationParser>(configuration->GetModellingConfigurationParser()));
        REQUIRE(instanceof<TraceWriter>(configuration->GetTraceWriter()));
    }

    SECTION("GenerateParameters Function Testing") {
        auto *computationParameters = generator->GenerateParameters();

        REQUIRE(instanceof<ComputationParameters>(computationParameters));
        REQUIRE(computationParameters->GetBoundaryLength() == 20);
        REQUIRE(computationParameters->GetRelaxedDT() == 0.9f);
        REQUIRE(computationParameters->GetSourceFrequency() == 20);
        REQUIRE(computationParameters->IsUsingWindow() == true);
        REQUIRE(computationParameters->GetLeftWindow() == 250);
        REQUIRE(computationParameters->GetRightWindow() == 250);
        REQUIRE(computationParameters->GetDepthWindow() == 500);
        REQUIRE(computationParameters->GetFrontWindow() == 0);
        REQUIRE(computationParameters->GetBackWindow() == 0);
        REQUIRE(computationParameters->GetEquationOrder() == SECOND);
        REQUIRE(computationParameters->GetApproximation() == ISOTROPIC);
        REQUIRE(computationParameters->GetPhysics() == ACOUSTIC);
        REQUIRE(computationParameters->GetBlockX() == 128);
        REQUIRE(computationParameters->GetBlockY() == 1);
        REQUIRE(computationParameters->GetBlockZ() == 16);

    }
    SECTION("GenerateRTMConfiguration Function Testing") {
        RTMEngineConfigurations *configuration = generator->GenerateRTMConfiguration(WRITE_PATH);

        REQUIRE(instanceof<ComputationKernel>(configuration->GetComputationKernel()));
        REQUIRE(instanceof<ModelHandler>(configuration->GetModelHandler()));
        REQUIRE(instanceof<SourceInjector>(configuration->GetSourceInjector()));
        REQUIRE(instanceof<BoundaryManager>(configuration->GetBoundaryManager()));
        REQUIRE(instanceof<ForwardCollector>(configuration->GetForwardCollector()));
        REQUIRE(instanceof<MigrationAccommodator>(configuration->GetMigrationAccommodator()));
        REQUIRE(instanceof<TraceManager>(configuration->GetTraceManager()));
    }

    SECTION("GenerateAgent Function Testing") {
        Agent *agent = generator->GenerateAgent();
        REQUIRE(instanceof<Agent>(agent));
    }

    SECTION("GenerateWriter Function Testing") {
        Writer *writer = generator->GenerateWriter();
        REQUIRE(instanceof<Writer>(writer));
    }
}

TEST_CASE("Generator Class Testing", "[Generators]") {
    TEST_CASE_GENERATOR();
}
