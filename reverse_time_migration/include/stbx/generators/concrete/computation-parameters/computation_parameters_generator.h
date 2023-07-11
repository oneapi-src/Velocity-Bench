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
// Created by amr-nasr on 12/12/2019.
//

#ifndef SEISMIC_TOOLBOX_GENERATORS_COMPUTATION_PARAMETERS_COMPUTATION_PARAMETERS_GENERATOR_H
#define SEISMIC_TOOLBOX_GENERATORS_COMPUTATION_PARAMETERS_COMPUTATION_PARAMETERS_GENERATOR_H

#include <operations/common/ComputationParameters.hpp>

#include <libraries/nlohmann/json.hpp>

operations::common::ComputationParameters *generate_parameters(nlohmann::json &map);

#endif // SEISMIC_TOOLBOX_GENERATORS_COMPUTATION_PARAMETERS_COMPUTATION_PARAMETERS_GENERATOR_H
