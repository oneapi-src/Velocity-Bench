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
// Created by zeyad-osama on 08/10/2020
//

///
/// @brief This should contain the main function that drives the
/// Seismic Engine execution.
///

#include <stbx/agents/Agents.h>
#include <stbx/writers/Writers.h>
#include <stbx/parsers/Parser.h>
#include <stbx/generators/Generator.hpp>
#include <stbx/utils/cmd/cmd_parser.hpp>
#include <chrono>

using namespace std;
using namespace stbx::parsers;
using namespace stbx::generators;
using namespace stbx::writers;
using namespace stbx::utils;
using namespace operations::dataunits;
using namespace operations::engines;

int main(int argc, char *argv[]) {
    try {
        std::chrono::steady_clock::time_point const tpStart(std::chrono::steady_clock::now());
        string parameter_file = WORKLOAD_PATH "/computation_parameters.json";
        string configuration_file = WORKLOAD_PATH "/engine_configuration.json";
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
        auto *engine_configuration = g->GenerateRTMConfiguration(write_path);
        g->GenerateCallbacks(write_path);
        auto *cbs = g->GetCallbackCollection();
        auto *engine = new RTMEngine(engine_configuration, cp, cbs);

        auto *agent = g->GenerateAgent();
        agent->AssignEngine(engine);
        agent->AssignArgs(argc, argv);
        MigrationData *md = agent->Execute();

        std::cout << "Finished executing" << std::endl;
        auto *writer = g->GenerateWriter();
        writer->AssignMigrationData(md);
        std::chrono::steady_clock::time_point const tpIOWriteStart(std::chrono::steady_clock::now());
        writer->Write(write_path); // This calls WriteSegy and WriteBinary, twice each (include/stbx/writers/interface/Writer.hpp)
        std::chrono::steady_clock::time_point const tpIOWriteEnd(std::chrono::steady_clock::now());
        double const dIOReadTime(engine->GetIOReadTime());

        delete engine_configuration;
        delete cbs;
        delete cp;
        delete engine;

        std::chrono::steady_clock::time_point const tpEnd(std::chrono::steady_clock::now());
        std::cout << "Total Execution Time: " << std::chrono::duration<double>( (tpEnd - tpStart) - (tpIOWriteEnd - tpIOWriteStart)).count() - dIOReadTime << " s" << std::endl;
    } catch (std::exception const &e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch (...) { 
        std::cerr << "Unknown exception caught" << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
