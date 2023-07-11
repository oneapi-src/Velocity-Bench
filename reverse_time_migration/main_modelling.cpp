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
// Created by amr-nasr on 12/11/2019.
//

///
/// @brief This should contain the main function that drives the
/// Seismic Engine Modelling execution.
///

#include <stbx/agents/Agents.h>
#include <stbx/writers/Writers.h>
#include <stbx/parsers/Parser.h>
#include <stbx/generators/Generator.hpp>
#include <stbx/utils/cmd/cmd_parser.hpp>

#include <operations/engines/concrete/ModellingEngine.hpp>

using namespace std;
using namespace stbx::parsers;
using namespace stbx::generators;
using namespace stbx::writers;
using namespace stbx::utils;
using namespace operations::dataunits;
using namespace operations::engines;

int main(int argc, char *argv[]) {
    try {
        string parameter_file = WORKLOAD_PATH "/computation_parameters.json";
        string configuration_file = WORKLOAD_PATH "/modelling_configuration.json";
        string callback_file = WORKLOAD_PATH "/callback_configuration.json";
        string pipeline = WORKLOAD_PATH "/pipeline.json";
        string write_path = WRITE_PATH;

        cout << "Starting Seismic Engine..." << endl;

        ifstream stream(write_path + "/timing_results.txt");

        parse_args(parameter_file, configuration_file, callback_file, pipeline,
                   write_path, argc, argv);

        Parser *p = Parser::GetInstance();
        p->RegisterFile(parameter_file);
        p->RegisterFile(configuration_file);
        p->RegisterFile(callback_file);
        p->RegisterFile(pipeline);
        p->BuildMap();

        auto *g = new Generator(p->GetMap());
        auto *cp = g->GenerateParameters();
        auto *engine_configuration = g->GenerateModellingEngineConfiguration(write_path);
        g->GenerateCallbacks(write_path);
        auto *cbs = g->GetCallbackCollection();
        auto *engine = new ModellingEngine(engine_configuration, cp, cbs);

        auto *agent = g->GenerateAgent();
        agent->AssignEngine(engine);
        agent->AssignArgs(argc, argv);
        agent->Execute();

        delete engine_configuration;
        delete cbs;
        delete cp;
        delete engine;
    } catch (std::exception const &e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch (...) { 
        std::cerr << "Unknown exception caught" << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
