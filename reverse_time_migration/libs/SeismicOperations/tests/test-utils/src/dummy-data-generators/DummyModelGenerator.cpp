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
// Created by marwan-elsafty on 04/02/2021.
//

#include <operations/test-utils/dummy-data-generators/DummyModelGenerator.hpp>

#include <operations/utils/io/write_utils.h>

using namespace std;

namespace operations {
    namespace testutils {

        float *generate_dummy_model(const std::string &aFileName) {
            int nx = 5;
            int ny = 5;
            int nz = 5;
            int nt = 1;
            float dx = 1.0f;
            float dy = 1.0f;
            float dz = 10.0f;
            float dt = 1.0;
            bool is_traces = false;
            string file_name = OPERATIONS_TEST_DATA_PATH "/" + aFileName + ".segy";
            int size = nx * ny * nz;

            auto data = new float[size];
            for (int i = 0; i < size; i++) {
                data[i] = 1;
            }

            operations::utils::io::write_segy(nx, ny, nz, nt,
                                              dx, dy, dz, dt,
                                              data, file_name, is_traces);
            return data;
        }
    } //namespace testutils
} //namespace operations
