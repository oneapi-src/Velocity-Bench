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
// Created by marwan-elsafty on 05/01/2021.
//

#include <operations/test-utils/dummy-data-generators/DummyConfigurationMapGenerator.hpp>

using namespace operations::configuration;
using json = nlohmann::json;

namespace operations {
    namespace testutils {

        JSONConfigurationMap *generate_average_case_configuration_map_wave() {
            return new JSONConfigurationMap(R"(
                {
                    "wave": {
                        "physics": "acoustic",
                        "approximation": "isotropic",
                        "equation-order": "first",
                        "grid-sampling": "uniform"
                    }
                }
            )"_json);
        }

    } //namespace testutils
} //namespace operations
