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
// Created by amr-nasr on 18/01/2020.
//

#include <operations/helpers/callbacks/concrete/NormWriter.h>

#include <operations/helpers/callbacks/interface/Extensions.hpp>

#include <cmath>
#include <sys/stat.h>

#define CAT_STR(a, b) (a + b)

using namespace std;
using namespace operations::helpers::callbacks;
using namespace operations::dataunits;
using namespace operations::common;


NormWriter::NormWriter(uint show_each, bool write_forward, bool write_backward,
                       bool write_reverse, const string &write_path) 
    : forward_norm_stream(nullptr)
    , reverse_norm_stream(nullptr)
    , backward_norm_stream(nullptr)
{
    this->show_each = show_each;
    this->write_forward = write_forward;
    this->write_backward = write_backward;
    this->write_reverse = write_reverse;
    this->write_path = write_path;
    mkdir(write_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    this->write_path = this->write_path + "/norm";
    mkdir(this->write_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (this->write_forward) {
        forward_norm_stream = new ofstream(this->write_path + "/forward_norm" + this->GetExtension());
    }
    if (this->write_reverse) {
        reverse_norm_stream = new ofstream(this->write_path + "/reverse_norm" + this->GetExtension());
    }
    if (this->write_backward) {
        backward_norm_stream =
                new ofstream(this->write_path + "/backward_norm" + this->GetExtension());
    }
}

NormWriter::~NormWriter() {
    if (this->write_forward) {
        delete forward_norm_stream;
    }
    if (this->write_reverse) {
        delete reverse_norm_stream;
    }
    if (this->write_backward) {
        delete backward_norm_stream;
    }
}

void NormWriter::BeforeInitialization(ComputationParameters *apParameters) {}

void NormWriter::AfterInitialization(GridBox *apGridBox) {}

void NormWriter::BeforeShotPreprocessing(TracesHolder *apTraces) {}

void NormWriter::AfterShotPreprocessing(TracesHolder *apTraces) {}

void NormWriter::BeforeForwardPropagation(GridBox *apGridBox) {}

void NormWriter::AfterForwardStep(GridBox *apGridBox, uint aTimeStep) {
    if (write_forward && aTimeStep % show_each == 0) {
        uint nx = apGridBox->GetActualWindowSize(X_AXIS);
        uint ny = apGridBox->GetActualWindowSize(Y_AXIS);
        uint nz = apGridBox->GetActualWindowSize(Z_AXIS);
        float norm = NormWriter::Solve(apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer(), nx, nz, ny);
        (*this->forward_norm_stream) << aTimeStep << "\t" << norm << endl;
    }
}

void NormWriter::BeforeBackwardPropagation(GridBox *apGridBox) {}

void NormWriter::AfterBackwardStep(GridBox *apGridBox, uint aTimeStep) {
    if (write_backward && aTimeStep % show_each == 0) {
        uint nx = apGridBox->GetActualWindowSize(X_AXIS);
        uint ny = apGridBox->GetActualWindowSize(Y_AXIS);
        uint nz = apGridBox->GetActualWindowSize(Z_AXIS);
        float norm = NormWriter::Solve(apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer(), nx, nz, ny);
        (*this->backward_norm_stream) << aTimeStep << "\t" << norm << endl;
    }
}

void NormWriter::AfterFetchStep(GridBox *apGridBox,
                                uint aTimeStep) {
    if (write_reverse && aTimeStep % show_each == 0) {
        uint nx = apGridBox->GetActualWindowSize(X_AXIS);
        uint ny = apGridBox->GetActualWindowSize(Y_AXIS);
        uint nz = apGridBox->GetActualWindowSize(Z_AXIS);
        float norm = NormWriter::Solve(apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer(), nx, nz, ny);
        (*this->reverse_norm_stream) << aTimeStep << "\t" << norm << endl;
    }
}

void NormWriter::BeforeShotStacking(
        GridBox *apGridBox, FrameBuffer<float> *apShotCorrelation) {}

void NormWriter::AfterShotStacking(
        GridBox *apGridBox, FrameBuffer<float> *apStackedShotCorrelation) {}

void NormWriter::AfterMigration(
        GridBox *apGridBox, FrameBuffer<float> *apStackedShotCorrelation) {}


float NormWriter::Solve(const float *apMatrix, uint nx, uint nz, uint ny) {
    float sum = 0;
    uint nx_nz = nx * nz;
    for (int iy = 0; iy < ny; iy++) {
        for (int iz = 0; iz < nz; iz++) {
            for (int ix = 0; ix < nx; ix++) {
                auto value = apMatrix[iy * nx_nz + nx * iz + ix];
                sum += (value * value);
            }
        }
    }
    return sqrtf(sum);
}

std::string NormWriter::GetExtension() {
    return OP_K_EXT_NRM;
}
