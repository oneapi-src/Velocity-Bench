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
// Created by mirna-moawad on 1/16/20.
//

#ifndef OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_FILE_HANDLER_H
#define OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_FILE_HANDLER_H

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace operations {
    namespace components {
        namespace helpers {

            void bin_file_save(const char *file_name, const float *data, const size_t &size);

            void bin_file_load(const char *file_name, float *data, const size_t &size);

        }//namespace helpers
    }//namespace components
}//namespace operations

#endif //OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_FILE_HANDLER_H
