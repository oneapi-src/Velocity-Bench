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
// Created by pancee on 1/3/21.
//

#include <thoth/utils/synthetic-generators/SyntheticModelGenerator.hpp>

#include <thoth/configurations/concrete/JSONConfigurationMap.hpp>
#include <thoth/configurations/interface/MapKeys.h>

#include <libraries/catch/catch.hpp>

#include <iostream>

using namespace thoth::generators;
using namespace thoth::dataunits;
using namespace thoth::configuration;
using json = nlohmann::json;


TEST_CASE("Test SyntheticModelGenerator GenerateModel()") {
//    json configuration_map;
//    configuration_map[IO_K_PROPERTIES][IO_K_READ_PATH] = TEST_DATA_PATH "/synthetic/meta-data.json";
//
//    SyntheticModelGenerator s(new JSONConfigurationMap(configuration_map));
//    s.AcquireConfiguration();
//    s.Generate();
//    s.BuildGather();
}
