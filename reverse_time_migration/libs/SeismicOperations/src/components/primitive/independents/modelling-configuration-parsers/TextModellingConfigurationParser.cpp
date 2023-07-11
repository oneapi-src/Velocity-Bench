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
// Created by amr-nasr on 13/11/2019.
//

#include <operations/components/independents/concrete/modelling-configuration-parsers/TextModellingConfigurationParser.hpp>


#include <libraries/nlohmann/json.hpp>

#include <iostream>

using namespace std;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;
using json = nlohmann::json;

TextModellingConfigurationParser::~TextModellingConfigurationParser() {}

void TextModellingConfigurationParser::AcquireConfiguration() {}

Point3D parse_point(json &map) {
    return Point3D(map["x"].get<uint>(),
                   map["y"].get<uint>(),
                   map["z"].get<uint>());
}

void print_point(Point3D *apPoint3D) {
    cout << "("
         << "x = " << apPoint3D->x << ", "
         << "y = " << apPoint3D->y << ", "
         << "z = " << apPoint3D->z
         << ")" << endl;
}

void PrintModellingConfiguration(ModellingConfiguration *apModellingConfiguration) {
    cout << "Simulation time : " << apModellingConfiguration->TotalTime << endl;
    cout << "Source point : ";
    print_point(&apModellingConfiguration->SourcePoint);
    cout << "Receivers start point : ";
    print_point(&apModellingConfiguration->ReceiversStart);
    cout << "Receivers increment values : ";
    print_point(&apModellingConfiguration->ReceiversIncrement);
    cout << "Receivers end point : ";
    print_point(&apModellingConfiguration->ReceiversEnd);
}

ModellingConfiguration
TextModellingConfigurationParser::ParseConfiguration(string filepath,
                                                     bool is_2D) {
    ifstream model_file(filepath);
    if (!model_file.is_open()) {
        std::cerr << "Couldn't open modelling parameters file '"
                  << filepath
                  << "'... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    ModellingConfiguration modelling_configuration;

    uint half_length = this->mpParameters->GetHalfLength();
    uint bound_length = this->mpParameters->GetBoundaryLength();

    std::ifstream in(filepath.c_str());
    json map;
    in >> map;
    json modelling_configuration_map = map["modelling-configuration"];
    json sub_map;

    sub_map = modelling_configuration_map["source"];
    if (!sub_map.empty()) {
        modelling_configuration.SourcePoint = parse_point(sub_map);
        modelling_configuration.SourcePoint.x += half_length + bound_length;
        if (!is_2D) {
            modelling_configuration.SourcePoint.y += half_length + bound_length;
        } else {
            modelling_configuration.SourcePoint.y = 0;
        }
        modelling_configuration.SourcePoint.z += half_length + bound_length;
    } else {
        modelling_configuration.SourcePoint.x
                = modelling_configuration.SourcePoint.y
                = modelling_configuration.SourcePoint.z
                = half_length + bound_length;
    }

    sub_map = modelling_configuration_map["receiver-start"];
    if (!sub_map.empty()) {
        modelling_configuration.ReceiversStart = parse_point(sub_map);
        modelling_configuration.ReceiversStart.x += half_length + bound_length;
        if (!is_2D) {
            modelling_configuration.ReceiversStart.y += half_length + bound_length;
        } else {
            modelling_configuration.ReceiversStart.y = 0;
        }
        modelling_configuration.ReceiversStart.z += half_length + bound_length;
    } else {
        modelling_configuration.ReceiversStart.x
                = modelling_configuration.ReceiversStart.y
                = modelling_configuration.ReceiversStart.z
                = half_length + bound_length;
    }

    sub_map = modelling_configuration_map["receiver-inc"];
    if (!sub_map.empty()) {
        modelling_configuration.ReceiversIncrement = parse_point(sub_map);
    } else {
        modelling_configuration.ReceiversIncrement.x
                = modelling_configuration.ReceiversIncrement.y
                = modelling_configuration.ReceiversIncrement.z
                = 0;
    }

    sub_map = modelling_configuration_map["receiver-end"];
    if (!sub_map.empty()) {
        modelling_configuration.ReceiversEnd = parse_point(sub_map);
        modelling_configuration.ReceiversEnd.x += half_length + bound_length;
        if (!is_2D) {
            modelling_configuration.ReceiversEnd.y += half_length + bound_length;
        } else {
            modelling_configuration.ReceiversEnd.y = 1;
        }
        modelling_configuration.ReceiversEnd.z += half_length + bound_length;
    } else {
        modelling_configuration.ReceiversEnd.x
                = modelling_configuration.ReceiversEnd.y
                = modelling_configuration.ReceiversEnd.z
                = half_length + bound_length;
    }

    if (modelling_configuration_map["simulation-time"].is_number_float()) {
        modelling_configuration.TotalTime = modelling_configuration_map["simulation-time"].get<float>();
    }
    PrintModellingConfiguration(&modelling_configuration);
    return modelling_configuration;
}

void TextModellingConfigurationParser::SetComputationParameters(
        ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void TextModellingConfigurationParser::SetGridBox(GridBox *apGridBox) {
    /// Not needed.
}