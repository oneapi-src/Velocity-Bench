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
// Created by zeyad-osama on 07/02/2021.
//

#include <operations/test-utils/dummy-data-generators/DummyTraceGenerator.hpp>

#include <operations/utils/io/write_utils.h>

using namespace std;

namespace operations {
    namespace testutils {

        float *generate_dummy_trace(const std::string &aFileName,
                                    dataunits::GridBox *apGridBox,
                                    int trace_stride_x,
                                    int trace_stride_y) {
            // One trace every n points in x
            uint nx = apGridBox->GetActualWindowSize(X_AXIS) / trace_stride_x;
            uint nz = apGridBox->GetActualWindowSize(Z_AXIS);
            uint ny = 1;
            uint nt = nz;

            float dz = apGridBox->GetCellDimensions(Z_AXIS);
            float dx = apGridBox->GetCellDimensions(Z_AXIS) * trace_stride_x;
            float dy = 1.0f;

            float dt = 1.0f;

            if (apGridBox->GetActualWindowSize(Y_AXIS) > 1) {
                // One trace every n points in y
                ny = apGridBox->GetActualWindowSize(Y_AXIS) / trace_stride_y;
                dy = apGridBox->GetCellDimensions(Y_AXIS) * trace_stride_y;
            }

            auto data = new float[nx * nz * ny];
            for (int i = 0; i < nx * nz * ny; i++) {
                data[i] = (float) rand() * 100 / RAND_MAX;
            }
            operations::utils::io::write_segy(nx, ny, nz, nt,
                                              dx, dy, dz, dt,
                                              data, aFileName, true);
            return data;
        }

    } //namespace testutils
} //namespace operations
