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
// Created by amr-nasr on 13/11/2019.
//

#include "operations/components/independents/concrete/trace-writers/BinaryTraceWriter.hpp"

#include <iostream>

using namespace std;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;

void BinaryTraceWriter::RecordTrace() {
    int x_inc = this->mReceiverIncrement.x == 0 ? 1 : this->mReceiverIncrement.x;
    int y_inc = this->mReceiverIncrement.y == 0 ? 1 : this->mReceiverIncrement.y;
    int z_inc = this->mReceiverIncrement.z == 0 ? 1 : this->mReceiverIncrement.z;
    int wnx = this->mpGridBox->GetActualWindowSize(X_AXIS);
    int wnz_wnx = this->mpGridBox->GetActualWindowSize(Z_AXIS) * wnx;
    float *pressure = this->mpGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    float *temp = new float[this->mReceiverEnd.x - this->mReceiverStart.x];
    for (int iz = this->mReceiverStart.z; iz < this->mReceiverEnd.z; iz += z_inc) {
        for (int iy = this->mReceiverStart.y; iy < this->mReceiverEnd.y; iy += y_inc) {
            int offset = iy * wnz_wnx + iz * wnx + this->mReceiverStart.x;
            Device::MemCpy(temp, pressure + offset, (this->mReceiverEnd.x -
                                                     this->mReceiverStart.x) * sizeof(float),
                           Device::COPY_DEVICE_TO_HOST);
            for (int ix = 0; ix < this->mReceiverEnd.x - this->mReceiverStart.x; ix += x_inc) {
                this->mpOutStream->write((char *) &temp[ix], sizeof(temp[ix]));
            }
        }
    }

    delete[] temp;
}