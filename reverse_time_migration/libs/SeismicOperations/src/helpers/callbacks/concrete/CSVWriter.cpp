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
// Created by amr-nasr on 31/12/2019.
//

#include <operations/helpers/callbacks/concrete/CSVWriter.h>

#include <operations/helpers/callbacks/interface/Extensions.hpp>
#include <operations/utils/io/write_utils.h>

#include <fstream>

using namespace std;
using namespace operations::helpers::callbacks;
using namespace operations::utils::io;


string CsvWriter::GetExtension() {
    return OP_K_EXT_CSV;
}

void CsvWriter::WriteCsv(float *mat, uint nz, uint nx, const string &filename,
                         uint start_x, uint start_z, uint end_x, uint end_z) {
    std::ofstream out(filename);
    out << (end_x - start_x) << "," << (end_z - start_z) << "\n";
    for (uint row = start_z; row < end_z; row++) {
        for (uint col = start_x; col < end_x; col++)
            out << mat[row * nx + col] << ',';
        out << '\n';
    }
}

void CsvWriter::WriteResult(uint nx, uint ny, uint nz, uint nt,
                            float dx, float dy, float dz, float dt, float *data, std::string filename, bool is_traces) {
    WriteCsv(data + (ny / 2) * nx * nz, nz, nx, filename, 0, 0, nx, nz);
}



