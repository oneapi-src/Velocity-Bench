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
// Created by marwan-elsafty on 21/01/2021.
//

#include <stbx/generators/primitive/ConfigurationsGenerator.hpp>

#include <operations/common/DataTypes.h>
#include <stbx/generators/common/Keys.hpp>

#include <iostream>
#include <string>

using namespace stbx::generators;
using namespace operations::configuration;


ConfigurationsGenerator::ConfigurationsGenerator(nlohmann::json &aMap) {
    this->mMap = aMap;
}

PHYSICS
ConfigurationsGenerator::GetPhysics() {
    nlohmann::json wave = this->mMap[K_WAVE];
    PHYSICS physics = ACOUSTIC;
    if (wave[K_PHYSICS].get<std::string>() == "acoustic") {
        physics = ACOUSTIC;
        std::cout << "Using Acoustic for physics" << std::endl;
    } else {
        std::cout << "Invalid value for physics key : supported values [ acoustic ]" << ::std::endl;
        std::cout << "Using Acoustic for physics" << std::endl;
    }
    return physics;
}

EQUATION_ORDER
ConfigurationsGenerator::GetEquationOrder() {
    nlohmann::json wave = this->mMap[K_WAVE];
    EQUATION_ORDER order = SECOND;
    if (wave[K_EQUATION_ORDER].get<std::string>() == "second") {
        order = SECOND;
        std::cout << "Using second order wave equation" << std::endl;
    } else if (wave[K_EQUATION_ORDER].get<std::string>() == "first") {
        order = FIRST;
        std::cout << "Using first order wave equation" << std::endl;
    } else {
        std::cout << "Invalid value for equation-order key : supported values [ second | first ]" << std::endl;
        std::cout << "Using second order wave equation" << std::endl;
    }
    return order;
}


GRID_SAMPLING
ConfigurationsGenerator::GetGridSampling() {
    nlohmann::json wave = this->mMap[K_WAVE];
    GRID_SAMPLING sampling = UNIFORM;
    if (wave[K_GRID_SAMPLING].get<std::string>() == "uniform") {
        sampling = UNIFORM;
        std::cout << "Using uniform grid sampling" << std::endl;
    } else {
        std::cout << "Invalid value for grid-sampling key : supported values [ uniform ]" << std::endl;
        std::cout << "Using uniform grid sampling" << std::endl;
    }
    return sampling;
}

APPROXIMATION
ConfigurationsGenerator::GetApproximation() {
    nlohmann::json wave = this->mMap[K_WAVE];
    APPROXIMATION approximation = ISOTROPIC;
    if (wave[K_APPROXIMATION].get<std::string>() == "isotropic") {
        approximation = ISOTROPIC;
        std::cout << "Using Isotropic as approximation" << std::endl;
    } else if (wave[K_APPROXIMATION].get<std::string>() == "vti") {
        approximation = VTI;
        std::cout << "Using VTI as approximation" << std::endl;
    } else if (wave[K_APPROXIMATION].get<std::string>() == "tti") {
        approximation = TTI;
        std::cout << "Using TTI as approximation" << std::endl;
    } else {
        std::cout << "Invalid value for approximation key : supported values [ isotropic | vti ]" << std::endl;
        std::cout << "Using Acoustic for Isotropic" << std::endl;
    }
    return approximation;
}


std::map<std::string, std::string>
ConfigurationsGenerator::GetModelFiles() {
    std::map<std::string, std::string> model_files;
    if (this->mMap[K_MODELS].empty()) {
        std::cout << "No entry for models key : a value providing the filename"
                     " to the file that contains the directories of model files(each in "
                     "a line) must be provided"
                  << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    nlohmann::json models = this->mMap[K_MODELS];
    if (!models[K_VELOCITY].is_null()) {
        model_files[K_VELOCITY] = models[K_VELOCITY].get<std::string>();
    }
    if (!models[K_DENSITY].is_null()) {
        model_files[K_DENSITY] = models[K_DENSITY].get<std::string>();
    }
    if (!models[K_DELTA].is_null()) {
        model_files[K_DELTA] = models[K_DELTA].get<std::string>();
    }
    if (!models[K_EPSILON].is_null()) {
        model_files[K_EPSILON] = models[K_EPSILON].get<std::string>();
    }
    if (!models[K_THETA].is_null()) {
        model_files[K_THETA] = models[K_THETA].get<std::string>();
    }
    if (!models[K_PHI].is_null()) {
        model_files[K_PHI] = models[K_PHI].get<std::string>();
    }
    std::cout << "The following model files were detected : " << std::endl;
    int index = 0;
    for (auto &model_file : model_files) {
        std::cout << "\t" << ++index << ". " << model_file.first << "\t: " << model_file.second << std::endl;
    }
    return model_files;
}

std::vector<std::string>
ConfigurationsGenerator::GetTraceFiles(RTMEngineConfigurations *aConfiguration) {
    nlohmann::json traces = this->mMap[K_TRACES];
    std::vector<std::string> traces_files;
    if (traces.empty()) {
        std::cout << "No entry for traces-list key : a value providing the filename"
                     " to the file that contains the directories of the traces "
                     "files(each in a line) must be provided.\n"
                  << "The file should have the start shot id in the first line, and the "
                     "ending shot id(exclusive) in the second line."
                  << "\nThe following lines should be the directories of the traces "
                     "file(each in a line"
                  << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    } else {
        std::cout << "Parsing minimum shot id..." << std::endl;
        if (traces[K_MIN].is_null()) {
            aConfiguration->SetSortMin(0);
        } else {
            try {
                aConfiguration->SetSortMin(traces[K_MIN].get<int>());
            }
            catch (std::invalid_argument &e) {
                aConfiguration->SetSortMin(0);
                std::cout << "Couldn't parse the minimum shot ID to process... Setting to 0" << std::endl;
            }
        }
        std::cout << "Parsing maximum shot id..." << std::endl;
        if (traces[K_MAX].is_null()) {
            aConfiguration->SetSortMax(UINT_MAX);
        } else {
            try {
                aConfiguration->SetSortMax(traces[K_MAX].get<int>());
            }
            catch (std::invalid_argument &e) {
                aConfiguration->SetSortMax(UINT_MAX);
                std::cout << "Couldn't parse the maximum shot ID to process... Setting to " << UINT_MAX << std::endl;
            }
        }

        std::cout << "Parsing sort type..." << std::endl;
        aConfiguration->SetSortKey("CSR");
        if (traces[K_SORT_TYPE].is_null()) {
            std::cout << "Couldn't parse  sort type... Setting to " << aConfiguration->GetSortKey() << std::endl;
        } else {
            aConfiguration->SetSortKey(traces[K_SORT_TYPE].get<std::string>());
        }
        std::cout << "The following trace files were detected : " << std::endl;
        for (int i = 0; i < traces[K_PATHS].size(); i++) {
            std::string path = traces[K_PATHS][i].get<std::string>();
            std::cout << "\t" << (i + 1) << ". " << path << std::endl;
            traces_files.push_back(path);
        }
        std::cout << "Minimum allowable ID : " << aConfiguration->GetSortMin() << std::endl;
        std::cout << "Maximum allowable ID : " << aConfiguration->GetSortMax() << std::endl;
        aConfiguration->SetTraceFiles(traces_files);
    }
    return traces_files;
}

std::string
ConfigurationsGenerator::GetModellingFile() {
    std::string file_name;
    if (this->mMap[K_MODELLING_FILE].is_null()) {
        std::cout << "No entry for trace-file key : a value providing the file_name "
                     "to the file that the recorded traces will be written into must "
                     "be provided"
                  << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
    file_name = this->mMap[K_MODELLING_FILE].get<std::string>();
    std::cout << "Modelling file : '" << file_name << "'" << std::endl;
    return file_name;
}

std::string
ConfigurationsGenerator::GetOutputFile() {
    std::string file_name;
    if (this->mMap[K_OUTPUT_FILE].is_null()) {
        std::cout << "No entry for trace-file key : a value providing the file_name "
                     "to the file that the recorded traces will be written into must "
                     "be provided"
                  << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
    file_name = this->mMap[K_OUTPUT_FILE].get<std::string>();
    std::cout << "Trace output file : '" << file_name << "'" << std::endl;
    return file_name;
}
