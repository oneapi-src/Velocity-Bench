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
// Created by ahmed-ayyad on 24/06/2020.
//

#include <stbx/utils/cmd/cmd_parser.hpp>

#include <iostream>
#include <cstdio>
#include <string>
#include <unistd.h>

using namespace std;

void print_help();

void stbx::utils::parse_args(string &parameter_file, string &configuration_file,
                             string &callback_file, string &pipeline,
                             string &write_path,
                             int argc, char **argv) {
    int len = string(WORKLOAD_PATH).length();
    string new_workload_path;

    int opt;
    while ((opt = getopt(argc, argv, ":m:p:c:w:e:h")) != -1) {
        switch (opt) {
            case 'm':
                new_workload_path = string(optarg);
                parameter_file.replace(parameter_file.begin(),
                                       parameter_file.begin() + len,
                                       new_workload_path);

                configuration_file.replace(configuration_file.begin(),
                                           configuration_file.begin() + len,
                                           new_workload_path);

                callback_file.replace(callback_file.begin(),
                                      callback_file.begin() + len,
                                      new_workload_path);

                pipeline.replace(pipeline.begin(),
                                 pipeline.begin() + len,
                                 new_workload_path);
                break;
            case 'p':
                parameter_file = string(optarg);
                break;
            case 'c':
                callback_file = string(optarg);
                break;
            case 'e':
                configuration_file = string(optarg);
                break;
            case 'w':
                write_path = string(optarg);
                break;
            case 'h':
                print_help();
                exit(EXIT_FAILURE);
            case ':':
                printf("Option needs a value\n");
                print_help();
                exit(EXIT_FAILURE);
            case '?':
                printf("Invalid option entered...\n");
                print_help();
                exit(EXIT_FAILURE);
        }
    }
    cout << "Using files:" << endl;
    cout << "\t- Computation parameters: " << parameter_file << endl;
    cout << "\t- Engine configurations: " << configuration_file << endl;
    cout << "\t- Callback configurations: " << callback_file << endl;
    cout << "\t- Write path: " << write_path << endl;
}

void print_help() {
    cout << "Usage:" << endl
         << "\t ./Engine <optional-flags>" << endl

         << "\nOptional flags:" << endl

         << "\n\t-m <workload-path>"
            "\n\t\tWorkloads configurations path."
            "\n\t\tDefault is \"./workloads/bp_model\"" << endl

         << "\n\t-p <computation-parameter-file-path>"
            "\n\t\tComputation parameter configurations path."
            "\n\t\tDefault is \"./workloads/bp_model/computation_parameters.json\"" << endl

         << "\n\t-e <engine-configurations-file-path>"
            "\n\t\tEngine configurations configurations path."
            "\n\t\tDefault is \"./workloads/bp_model/engine_configuration.json\"" << endl

         << "\n\t-c <callbacks-configurations-file-path>"
            "\n\t\tCallbacks configurations file path."
            "\n\t\tDefault is \"./workloads/bp_model/callback_configuration.json\"" << endl

         << "\n\t-w <write path>"
            "\n\t\tResults write path."
            "\n\t\tDefault is \"./results\"" << endl

         << "\n\t-h"
            "\n\t\tHelp window" << endl;
}

