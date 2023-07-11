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

#include <operations/helpers/callbacks/concrete/ImageWriter.h>

#include <operations/helpers/callbacks/interface/Extensions.hpp>

#include <visualization/visualization.h>

using namespace std;
using namespace operations::helpers::callbacks;


string ImageWriter::GetExtension() {
    return OP_K_EXT_IMG;
}

void ImageWriter::WriteResult(uint nx, uint ny, uint nz, uint nt,
                              float dx, float dy, float dz, float dt, float *data, std::string filename,
                              bool is_traces) {
    GridToPNG(data + (ny / 2) * nx * nz, nz, nx, mPercentile, (char *) filename.c_str());
}