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
// Created by marwan on 11/01/2021.
//

#ifndef OPERATIONS_LIB_TEST_UTILS_DUMMY_DATA_GENERATORS_DUMMY_CONFIGURATION_MAP_GENERATOR_HPP
#define OPERATIONS_LIB_TEST_UTILS_DUMMY_DATA_GENERATORS_DUMMY_CONFIGURATION_MAP_GENERATOR_HPP

#include <operations/configurations/concrete/JSONConfigurationMap.hpp>

#include <libraries/nlohmann/json.hpp>

namespace operations {
    namespace testutils {

        configuration::JSONConfigurationMap *generate_average_case_configuration_map_wave();

    } //namespace testutils
} //namespace operations

#endif //OPERATIONS_LIB_TEST_UTILS_DUMMY_DATA_GENERATORS_DUMMY_CONFIGURATION_MAP_GENERATOR_HPP
