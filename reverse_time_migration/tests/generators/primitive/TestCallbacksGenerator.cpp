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
// Created by marwan-elsafty on 08/02/2021.
//
#include <stbx/generators/primitive/CallbacksGenerator.hpp>

#include <operations/helpers/callbacks/interface/Callback.hpp>
#include <operations/helpers/callbacks/primitive/CallbackCollection.hpp>
#include <operations/helpers/callbacks/concrete/BinaryWriter.h>
#include <operations/helpers/callbacks/concrete/SUWriter.h>
#include <operations/helpers/callbacks/concrete/SegyWriter.h>
#include <operations/helpers/callbacks/concrete/CSVWriter.h>
#include <operations/helpers/callbacks/concrete/ImageWriter.h>

#if defined  (USING_OMP) || (USING_DPCPP)

#include <operations/helpers/callbacks/concrete/NormWriter.h>

#endif

#include <stbx/test-utils/utils.h>

#include <libraries/catch/catch.hpp>
#include <libraries/nlohmann/json.hpp>

#include <vector>

using namespace stbx::generators;
using namespace stbx::testutils;

using namespace operations::common;
using namespace operations::configuration;
using namespace operations::components;
using namespace operations::exceptions;
using namespace operations::helpers;

/**
 * @brief Test all generated Callback WritersBooleans
 *
 * @note Get the Callbacks collection vector, pop the callbacks from
 * it the opposite way we pushed them in CallbacksGenerator and check
 * the type of each callback popped.
 */
void TEST_CASE_CALLBACKS_GENERATOR() {
    nlohmann::json map = R"(
{
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
}
            )"_json;

    auto *callbacks_generator = new CallbacksGenerator(WRITE_PATH, map);

    SECTION("CallbacksGenerator") {
        REQUIRE(instanceof<CallbacksGenerator>(callbacks_generator));
    }

    SECTION("GenerateCallbacks") {
        auto *callback_collection = callbacks_generator->GenerateCallbacks();
        REQUIRE(instanceof<callbacks::CallbackCollection>(callback_collection));

        std::vector<callbacks::Callback *> callbacks = callback_collection->GetCallbacks();

        REQUIRE(dynamic_cast<callbacks::BinaryWriter *>(callbacks.back()) != nullptr);
        callbacks.pop_back();

        REQUIRE(dynamic_cast<callbacks::SuWriter *>(callbacks.back()) != nullptr);
        callbacks.pop_back();

        REQUIRE(dynamic_cast<callbacks::SegyWriter *>(callbacks.back()) != nullptr);
        callbacks.pop_back();

        REQUIRE(dynamic_cast<callbacks::CsvWriter *>(callbacks.back()) != nullptr);
        callbacks.pop_back();

#if defined  (USING_OMP) || (USING_DPCPP)
        REQUIRE(dynamic_cast<callbacks::NormWriter *>(callbacks.back()) != nullptr);
        callbacks.pop_back();
#endif

#ifdef USE_OpenCV
        REQUIRE(dynamic_cast<callbacks::ImageWriter *>(callbacks.back()) != nullptr);
        callbacks.pop_back();
#endif
    }
}

TEST_CASE("CallbacksGenerator Class", "[Generator],[CallbacksGenerator]") {
    TEST_CASE_CALLBACKS_GENERATOR();
}
