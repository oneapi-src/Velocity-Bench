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

#include <operations/helpers/callbacks/concrete/BinaryWriter.h>

#include <operations/helpers/callbacks/interface/Extensions.hpp>
#include <operations/utils/io/write_utils.h>

using namespace std;
using namespace operations::helpers::callbacks;
using namespace operations::utils::io;


string BinaryWriter::GetExtension() {
    return OP_K_EXT_BIN;
}

void BinaryWriter::WriteResult(uint nx, uint ny, uint nz, uint nt,
                               float dx, float dy, float dz, float dt, float *data, std::string filename,
                               bool is_traces) {
    write_binary(data, nx, nz, filename.c_str());
}