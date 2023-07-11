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
// Created by marwan-elsafty on 18/01/2021.
//

#include <stbx/generators/primitive/ComponentsGenerator.hpp>

#include <stbx/generators/common/Keys.hpp>

using namespace std;
using namespace stbx::generators;
using namespace operations::common;
using namespace operations::configuration;
using namespace operations::components;
using namespace operations::exceptions;


ComponentsGenerator::ComponentsGenerator(const nlohmann::json &aMap,
                                         EQUATION_ORDER aOrder,
                                         GRID_SAMPLING aSampling,
                                         APPROXIMATION aApproximation) 
{
    this->mMap = aMap;
    this->mOrder = aOrder;
    this->mSampling = aSampling;
    this->mApproximation = aApproximation;
    this->m_ModellingConfigurationParser = nullptr;
}

ComputationKernel *
ComponentsGenerator::GenerateComputationKernel() {
    /* First order checking to know what if exit or not. */
    this->CheckFirstOrder();

    auto map = this->TruncateMap(K_COMPUTATION_KERNEL);

    if (this->mOrder == FIRST && this->mSampling == UNIFORM) {
        std:;cout << "Staggered computational kernel is not available" << std::endl;
        assert(0);
    } else if (this->mOrder == SECOND && this->mSampling == UNIFORM) {
        switch (this->mApproximation) {
            case ISOTROPIC:
                return new SecondOrderComputationKernel(map);
        }
    }

    std::cout << "No entry for wave->physics to identify Computation Kernel..." << std::endl; // We should not reach here
    std::cout << "Terminating..." << std::endl;
    exit(EXIT_FAILURE);
}

ModelHandler *
ComponentsGenerator::GenerateModelHandler() {
    /* First order checking to know what if exit or not. */
    this->CheckFirstOrder();

    if (this->mMap[K_MODEL_HANDLER].empty()) {
        std::cout << "No entry for model-handler key : supported values [ homogenous | segy ]" << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    string type = this->mMap[K_MODEL_HANDLER][OP_K_TYPE].get<string>();
    JSONConfigurationMap *model_handler_map = this->TruncateMap(K_MODEL_HANDLER);
    ModelHandler *model_handler;

    if (type == "synthetic") {
        model_handler = new SyntheticModelHandler(model_handler_map);
    } else if (type == "segy") {
        model_handler = new SeismicModelHandler(model_handler_map);
    } else {
        std::cout << "Invalid value for model-handler key : supported values [ homogenous | segy ]" << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
    return model_handler;
}

SourceInjector *
ComponentsGenerator::GenerateSourceInjector() {
    /* First order checking to know what if exit or not. */
    this->CheckFirstOrder();

    if (this->mMap[K_SOURCE_INJECTOR].empty()) {
        std::cout << "No entry for source-injector key : supported values [ ricker ]" << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    auto type = this->mMap[K_SOURCE_INJECTOR][OP_K_TYPE].get<string>();
    auto map = this->TruncateMap(K_SOURCE_INJECTOR);
    SourceInjector *source_injector;

    if (type == "ricker") {
        source_injector = new RickerSourceInjector(map);
    } else {
        std::cout << "Invalid value for source-injector key : supported values [ ricker ]" << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
    return source_injector;
}

BoundaryManager *
ComponentsGenerator::GenerateBoundaryManager() {
    /* First order checking to know what if exit or not. */
    this->CheckFirstOrder();

    if (this->mMap[K_BOUNDARY_MANAGER].empty()) {
        std::cout << "No entry for boundary-manager key : supported values " << K_SUPPORTED_VALUES_BOUNDARY_MANAGER
                  << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    auto type = this->mMap[K_BOUNDARY_MANAGER][OP_K_TYPE].get<string>();
    auto map = this->TruncateMap(K_BOUNDARY_MANAGER);
    BoundaryManager *boundary_manager = nullptr;

    if (type == "none") {
        boundary_manager = new NoBoundaryManager(map);
    } else {
        std::cout << "Invalid value for boundary-manager key : supported values " << K_SUPPORTED_VALUES_BOUNDARY_MANAGER
                  << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
    return boundary_manager;
}

ForwardCollector *
ComponentsGenerator::GenerateForwardCollector(const string &write_path) {
    /* First order checking to know what if exit or not. */
    this->CheckFirstOrder();

    if (this->mMap[K_FORWARD_COLLECTOR].empty()) {
        std::cout << "No entry for forward-collector key : supported values " << K_SUPPORTED_VALUES_FORWARD_COLLECTOR
                  << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    auto type = this->mMap[K_FORWARD_COLLECTOR][OP_K_TYPE].get<string>();
    auto map = this->TruncateMap(K_FORWARD_COLLECTOR);

    map->WriteValue(OP_K_PROPRIETIES, OP_K_WRITE_PATH, write_path);
    ForwardCollector *forward_collector = nullptr;

    if (this->mOrder == FIRST && this->mSampling == UNIFORM) {
        if (type == "two") {
            forward_collector = new TwoPropagation(map);
        } else if (type == "three") {
            forward_collector = new ReversePropagation(map);
        } else {
            std::cout << "Invalid value for forward-collector key : supported values "
                      << K_SUPPORTED_VALUES_FORWARD_COLLECTOR
                      << std::endl;
            std::cout << "Terminating..." << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (this->mOrder == SECOND && this->mSampling == UNIFORM) {
        if (type == "three") {
            forward_collector = new ReversePropagation(map);
        } else if (type == "two") {
            forward_collector = new TwoPropagation(map);
        } else {
            std::cout << "Invalid value for forward-collector key : supported values "
                      << K_SUPPORTED_VALUES_FORWARD_COLLECTOR << std::endl;
            std::cout << "Terminating..." << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    return forward_collector;
}


MigrationAccommodator *
ComponentsGenerator::GenerateMigrationAccommodator() {
    /* First order checking to know what if exit or not. */
    this->CheckFirstOrder();

    if (this->mMap[K_MIGRATION_ACCOMMODATOR].empty()) {
        std::cout << "No entry for migration-accommodator key : supported values [ "
                     "cross-correlation ]"
                  << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    auto type = this->mMap[K_MIGRATION_ACCOMMODATOR][OP_K_TYPE].get<string>();
    auto map = this->TruncateMap(K_MIGRATION_ACCOMMODATOR);
    MigrationAccommodator *correlation_kernel = nullptr;

    if (type == "cross-correlation") {
        correlation_kernel = new CrossCorrelationKernel(map);
    } else {
        std::cout << "Invalid value for migration-accommodator key : supported values [ "
                     "cross-correlation ]"
                  << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
    return correlation_kernel;
}

TraceManager *
ComponentsGenerator::GenerateTraceManager() {
    /* First order checking to know what if exit or not. */
    this->CheckFirstOrder();

    if (this->mMap[K_TRACE_MANAGER].empty()) {
        std::cout << "No entry for trace-manager key : supported values [ binary | segy ]" << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    auto type = this->mMap[K_TRACE_MANAGER][OP_K_TYPE].get<string>();
    auto map = this->TruncateMap(K_TRACE_MANAGER);
    TraceManager *trace_manager = nullptr;

    if (type == "binary") {
        trace_manager = new BinaryTraceManager(map);
    } else if (type == "segy") {
        trace_manager = new SeismicTraceManager(map);
    } else {
        std::cout << "Invalid value for trace-manager key : supported values [ binary | segy ]" << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
    return trace_manager;
}

void ComponentsGenerator::GenerateModellingConfigurationParser() 
{
    /* First order checking to know what if exit or not. */
    this->CheckFirstOrder();

    if (this->mMap[K_MODELLING_CONFIGURATION_PARSER].empty()) {
        std::cout << "No entry for modelling-configuration-parser key : supported values [ text ]" << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    auto type = this->mMap[K_MODELLING_CONFIGURATION_PARSER][OP_K_TYPE].get<string>();
    auto map = this->TruncateMap(K_MODELLING_CONFIGURATION_PARSER);

    if (type == "text") {
        m_ModellingConfigurationParser = new TextModellingConfigurationParser();
        assert(m_ModellingConfigurationParser != nullptr);
    } else {
        std::cout << "Invalid value for modelling-configuration-parser key : supported values [ text ]" << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
    delete map;
}

TraceWriter *
ComponentsGenerator::GenerateTraceWriter() {
    /* First order checking to know what if exit or not. */
    this->CheckFirstOrder();

    if (this->mMap[K_TRACE_WRITER].empty()) {
        std::cout << "No entry for trace-writer key : supported values [ binary ]" << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    auto type = this->mMap[K_TRACE_WRITER][OP_K_TYPE].get<string>();
    JSONConfigurationMap *trace_writer_map = this->TruncateMap(K_TRACE_WRITER);
    TraceWriter *trace_writer;

    if (type == "binary") {
        trace_writer = new BinaryTraceWriter(trace_writer_map);
    } else {
        std::cout << "Invalid value for trace-writer key : supported values [ binary ]" << std::endl;
        std::cout << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
    return trace_writer;
}

JSONConfigurationMap *
ComponentsGenerator::TruncateMap(const string &aComponentName) {
    auto map = this->GetWaveMap();
    map.merge_patch(this->mMap[aComponentName]);
    return new JSONConfigurationMap(map);
}

nlohmann::json ComponentsGenerator::GetWaveMap() {
    nlohmann::json map;
    map[K_WAVE][K_APPROXIMATION] = this->mApproximation;
    map[K_WAVE][K_SAMPLING] = this->mSampling;
    map[K_WAVE][K_EQUATION_ORDER] = this->mOrder;
    return map;
}

void ComponentsGenerator::CheckFirstOrder() {
    if (this->mOrder == FIRST && this->mSampling == UNIFORM) {
#if defined(USING_DPCPP)
        std::cout << "First order not supported yet..." << std::endl;
        exit(EXIT_FAILURE);
#endif
    }
}
