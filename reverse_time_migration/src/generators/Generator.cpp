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
// Created by marwan-elsafty on 23/11/2020.
//

#include <stbx/generators/Generator.hpp>

#include <stbx/generators/common/Keys.hpp>
#include <stbx/generators/primitive/ConfigurationsGenerator.hpp>
#include <stbx/generators/primitive/CallbacksGenerator.hpp>
#include <stbx/generators/concrete/computation-parameters/computation_parameters_generator.h>
#include <stbx/generators/primitive/ComponentsGenerator.hpp>

#include <iostream>
#include <string>

using namespace std;

using namespace stbx::agents;
using namespace stbx::writers;
using namespace stbx::generators;

using namespace operations::common;
using namespace operations::helpers::callbacks;
using namespace operations::configuration;
using namespace operations::components;
using namespace operations::exceptions;


Generator::Generator(const nlohmann::json &mMap) {
    this->mMap = mMap;
    this->mConfigurationsGenerator = new ConfigurationsGenerator(this->mMap);
    this->mOrder = this->mConfigurationsGenerator->GetEquationOrder();
    this->mSampling = this->mConfigurationsGenerator->GetGridSampling();
    this->mApproximation = this->mConfigurationsGenerator->GetApproximation();
    this->m_CallbacksGenerator = nullptr;
}

Generator::~Generator()
{
    if (mConfigurationsGenerator != nullptr)
        delete mConfigurationsGenerator;

    if (m_CallbacksGenerator != nullptr)
        delete m_CallbacksGenerator;
}

void Generator::GenerateCallbacks(const string &aWritePath) 
{
    nlohmann::json callbacks_map = this->mMap[K_CALLBACKS];
    m_CallbacksGenerator = new CallbacksGenerator(aWritePath, callbacks_map);
    assert(m_CallbacksGenerator != nullptr);
    m_CallbacksGenerator->GenerateCallbacks();
}

ModellingEngineConfigurations *
Generator::GenerateModellingEngineConfiguration(const string &aWritePath) {
    auto *configuration = new ModellingEngineConfigurations();

    configuration->SetModelFiles(this->mConfigurationsGenerator->GetModelFiles());
    configuration->SetModellingConfigurationFile(this->mConfigurationsGenerator->GetModellingFile());
    configuration->SetTraceFiles(this->mConfigurationsGenerator->GetOutputFile());

    auto g = new ComponentsGenerator(this->mMap[K_COMPONENTS],
                                     this->mOrder,
                                     this->mSampling,
                                     this->mApproximation);

    if ((this->mOrder == FIRST) || (this->mOrder == SECOND) && (this->mSampling == UNIFORM)) {
        configuration->SetComputationKernel(g->GenerateComputationKernel());
        configuration->SetModelHandler(g->GenerateModelHandler());
        configuration->SetSourceInjector(g->GenerateSourceInjector());
        configuration->SetBoundaryManager(g->GenerateBoundaryManager());
        g->GenerateModellingConfigurationParser();
        configuration->SetModellingConfigurationParser(g->GetModellingConfigurationParser());
        configuration->SetTraceWriter(g->GenerateTraceWriter());
    } else {
        cout << "Unsupported settings" << std::endl;
        exit(EXIT_FAILURE);
    }
    delete g;
    return configuration;
}

ComputationParameters *
Generator::GenerateParameters() {
    return generate_parameters(this->mMap);
}

RTMEngineConfigurations *
Generator::GenerateRTMConfiguration(const string &aWritePath) {
    auto *configuration = new RTMEngineConfigurations();

    cout << "Reading model files..." << std::endl;
    configuration->SetModelFiles(this->mConfigurationsGenerator->GetModelFiles());

    cout << "Reading trace files..." << std::endl;
    this->mConfigurationsGenerator->GetTraceFiles(configuration);

    auto g = new ComponentsGenerator(this->mMap[K_COMPONENTS],
                                     this->mOrder,
                                     this->mSampling,
                                     this->mApproximation);

    if ((this->mOrder == FIRST) || (this->mOrder == SECOND) && (this->mSampling == UNIFORM)) {
        configuration->SetComputationKernel(g->GenerateComputationKernel());
        configuration->SetModelHandler(g->GenerateModelHandler());
        configuration->SetSourceInjector(g->GenerateSourceInjector());
        configuration->SetBoundaryManager(g->GenerateBoundaryManager());
        configuration->SetForwardCollector(g->GenerateForwardCollector(aWritePath));
        configuration->SetMigrationAccommodator(g->GenerateMigrationAccommodator());
        configuration->SetTraceManager(g->GenerateTraceManager());
    } else {
        cout << "Unsupported settings" << std::endl;
        exit(EXIT_FAILURE);
    }
    delete g;
    return configuration;
}

Agent *Generator::GenerateAgent() {
    nlohmann::json agents_map = this->mMap["pipeline"]["agent"];

    Agent *agent;
    if (agents_map[OP_K_TYPE].get<string>() == "normal") {
        agent = new NormalAgent();
        cout << "using single Agent" << std::endl;
    }
#if defined(USING_MPI)
        else if (agents_map[OP_K_TYPE].get<string>() == "mpi-static-server") {
            cout << "Using MPI Shot Distribution: "
                    "\n\tDistribution Type: Static With Server" << std::endl;
            agent = new StaticServerAgent();
        } else if (agents_map[OP_K_TYPE].get<string>() == "mpi-static-serverless") {
            cout << "Using MPI Shot Distribution: "
                    "\n\tDistribution Type: Static Without Server" << std::endl;
            agent = new StaticServerlessAgent();
        } else if (agents_map[OP_K_TYPE].get<string>() == "mpi-dynamic-server") {
            cout << "Using MPI Shot Distribution: "
                    "\n\tDistribution Type: Dynamic With Server" << std::endl;
            agent = new DynamicServerAgent();
        } else if (agents_map[OP_K_TYPE].get<string>() == "mpi-dynamic-serverless") {
            cout << "Using MPI Shot Distribution:"
                    "\n\tDistribution Type: Dynamic Without Server" << std::endl;
            agent = new DynamicServerlessAgent();
        }
#endif
    else {
        throw new UndefinedException();
    }
    return agent;
}

Writer *Generator::GenerateWriter() {
    nlohmann::json migration_accommodator_map = this->mMap[K_COMPONENTS][K_MIGRATION_ACCOMMODATOR];

    Writer *writer;
    if (migration_accommodator_map[OP_K_TYPE].get<string>() == "cross-correlation") {
        writer = new NormalWriter();
    } else if (migration_accommodator_map[OP_K_TYPE].get<string>() == "adcig") {
        writer = new ADCIGWriter();
    } else {
        throw new UndefinedException();
    }
    return writer;
}


