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
// Created by amr-nasr on 25/01/2020.
//

#ifndef SEISMIC_TOOLBOX_UTILS_CMD_CMD_PARSER_UTILS_H
#define SEISMIC_TOOLBOX_UTILS_CMD_CMD_PARSER_UTILS_H

#include <string>

namespace stbx {
    namespace utils {

        void parse_args(std::string &parameter_file,
                        std::string &configuration_file,
                        std::string &callback_file,
                        std::string &pipeline,
                        std::string &write_path,
                        int argc, char **argv);

    }//namespace utils
}//namespace stbx

#endif //SEISMIC_TOOLBOX_UTILS_CMD_CMD_PARSER_UTILS_H