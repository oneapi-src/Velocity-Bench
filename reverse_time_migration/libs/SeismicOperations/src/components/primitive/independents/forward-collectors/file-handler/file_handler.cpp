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

#include "operations/components/independents/concrete/forward-collectors/file-handler/file_handler.h"

void operations::components::helpers::bin_file_save(
        const char *file_name, const float *data, const size_t &size) {
    std::ofstream stream(file_name, std::ios::out | std::ios::binary);
    if (!stream.is_open()) {
        exit(EXIT_FAILURE);
    }
    stream.write(reinterpret_cast<const char *>(data), size * sizeof(float));
    stream.close();
}

void operations::components::helpers::bin_file_load(
        const char *file_name, float *data, const size_t &size) {
    std::ifstream stream(file_name, std::ios::binary | std::ios::in);
    if (!stream.is_open()) {
        exit(EXIT_FAILURE);
    }
    stream.read(reinterpret_cast<char *>(data), std::streamsize(size * sizeof(float)));
    stream.close();
}