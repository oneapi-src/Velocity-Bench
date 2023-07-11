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

#include <stbx/generators/primitive/ComputationParametersGetter.hpp>

#include <libraries/catch/catch.hpp>
#include <libraries/nlohmann/json.hpp>

using namespace stbx::generators;


void TEST_CASE_BASE_COMPUTATION_PARAMETERS_GETTER() {
    nlohmann::json map = R"(
        {
    "stencil-order": 8,
    "boundary-length": 20,
    "source-frequency": 20,
    "isotropic-radius": 5,
    "dt-relax": 0.9,
    "algorithm": "cpu",
    "Device": "none",
    "cache-blocking": {
      "block-x": 5500,
      "block-z": 55,
      "block-y": 1,
      "cor-block": 256
    }
}
 )"_json;

    auto computation_parameters_getter = new ComputationParametersGetter(map);

    SECTION("GetBoundaryLength Testing") {
        REQUIRE(computation_parameters_getter->GetBoundaryLength() == 20);
    }
    SECTION("GetSourceFrequency Testing") {
        REQUIRE(computation_parameters_getter->GetSourceFrequency() == 20.0);
    }
    SECTION("GetDTRelaxed Testing") {
        REQUIRE(computation_parameters_getter->GetDTRelaxed() == 0.9f);
    }
    SECTION("GetBlock Testing") {
        REQUIRE(computation_parameters_getter->GetBlock("x") == 5500);
        REQUIRE(computation_parameters_getter->GetBlock("y") == 1);
        REQUIRE(computation_parameters_getter->GetBlock("z") == 55);
    }
    SECTION("GetIsotropicCircle Testing") {
        REQUIRE(computation_parameters_getter->GetIsotropicCircle() == 5);
    }
    delete computation_parameters_getter;
}

void TEST_CASE_STENCIL_ORDER_GETTER() {
    SECTION("stencil-order = 2") {
        nlohmann::json map = R"(
        {"stencil-order": 2}
        )"_json;
        auto *computation_parameters_getter = new ComputationParametersGetter(map);

        StencilOrder so = computation_parameters_getter->GetStencilOrder();
        REQUIRE(so.order == 2);
        REQUIRE(so.half_length == O_2);
    }
    SECTION("stencil-order = 4") {
        nlohmann::json map = R"(
        {"stencil-order": 4}
        )"_json;
        auto *computation_parameters_getter = new ComputationParametersGetter(map);

        StencilOrder so = computation_parameters_getter->GetStencilOrder();
        REQUIRE(so.order == 4);
        REQUIRE(so.half_length == O_4);
    }
    SECTION("stencil-order = 8") {
        nlohmann::json map = R"(
        {"stencil-order": 8}
        )"_json;
        auto *computation_parameters_getter = new ComputationParametersGetter(map);

        StencilOrder so = computation_parameters_getter->GetStencilOrder();
        REQUIRE(so.order == 8);
        REQUIRE(so.half_length == O_8);
    }
    SECTION("stencil-order = 12") {
        nlohmann::json map = R"(
        {"stencil-order": 12}
        )"_json;
        auto *computation_parameters_getter = new ComputationParametersGetter(map);

        StencilOrder so = computation_parameters_getter->GetStencilOrder();
        REQUIRE(so.order == 12);
        REQUIRE(so.half_length == O_12);
    }
    SECTION("stencil-order = 16") {
        nlohmann::json map = R"(
        {"stencil-order": 16}
        )"_json;
        auto *computation_parameters_getter = new ComputationParametersGetter(map);

        StencilOrder so = computation_parameters_getter->GetStencilOrder();
        REQUIRE(so.order == 16);
        REQUIRE(so.half_length == O_16);
    }
}

void TEST_CASE_WINDOW_GETTER() {
    /// Average test cases parameter
    nlohmann::json average_map = R"(
    {
     "window": {
      "enable": true,
      "left": 250,
      "right": 250,
      "depth": 500,
      "front": 0,
      "back": 0}
    }
    )"_json;

    /// Minimum test cases parameter
    nlohmann::json min_map = R"(
    {
     "window": {
      "enable": true,
      "left": -10,
      "right": -10,
      "depth": -10,
      "front": -10,
      "back": -10}
    }
    )"_json;

    /// Null test cases parameter
    nlohmann::json null_map = R"(
    {
     "window": {
      "enable": false}
    }
    )"_json;

    /// Missing test cases parameter
    nlohmann::json missing_map = R"(
    {
     "window": {
      "enable": true}
    }
    )"_json;

    SECTION("Average Case") {
        auto computation_parameters_getter = new ComputationParametersGetter(average_map);
        auto w = computation_parameters_getter->GetWindow();
        REQUIRE(w.use_window == 1);
        REQUIRE(w.left_win == 250);
        REQUIRE(w.right_win == 250);
        REQUIRE(w.depth_win == 500);
        REQUIRE(w.front_win == 0);
        REQUIRE(w.back_win == 0);
        delete computation_parameters_getter;
    }
    SECTION("Minimum Case") {
        auto computation_parameters_getter = new ComputationParametersGetter(min_map);
        auto w = computation_parameters_getter->GetWindow();
        REQUIRE(w.use_window == 1);
        REQUIRE(w.left_win == DEF_VAL);
        REQUIRE(w.right_win == DEF_VAL);
        REQUIRE(w.depth_win == DEF_VAL);
        REQUIRE(w.front_win == DEF_VAL);
        REQUIRE(w.back_win == DEF_VAL);
        delete computation_parameters_getter;
    }

    SECTION("Null Case") {
        auto computation_parameters_getter = new ComputationParametersGetter(null_map);
        auto w = computation_parameters_getter->GetWindow();
        REQUIRE(w.use_window == 0);
        REQUIRE(w.left_win == DEF_VAL);
        REQUIRE(w.right_win == DEF_VAL);
        REQUIRE(w.depth_win == DEF_VAL);
        REQUIRE(w.front_win == DEF_VAL);
        REQUIRE(w.back_win == DEF_VAL);
        delete computation_parameters_getter;
    }

    SECTION("missing values") {
        auto computation_parameters_getter = new ComputationParametersGetter(missing_map);
        auto w = computation_parameters_getter->GetWindow();
        REQUIRE(w.use_window == 1);
        REQUIRE(w.left_win == DEF_VAL);
        REQUIRE(w.right_win == DEF_VAL);
        REQUIRE(w.depth_win == DEF_VAL);
        REQUIRE(w.front_win == DEF_VAL);
        REQUIRE(w.back_win == DEF_VAL);
        delete computation_parameters_getter;
    }
}

TEST_CASE("ComputationParametersGetter Class - Base",
          "[Generator],[ComputationParametersGetter]") {
    TEST_CASE_BASE_COMPUTATION_PARAMETERS_GETTER();
}

TEST_CASE("ComputationParametersGetter Class - Stencil Order",
          "[Generator],[ComputationParametersGetter]") {
    TEST_CASE_STENCIL_ORDER_GETTER();
}

TEST_CASE("ComputationParametersGetter Class - Window Getter",
          "[Generator],[ComputationParametersGetter]") {
    TEST_CASE_WINDOW_GETTER();
}
