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
// Created by mirna-moawad on 22/10/2019.
//

#include "operations/components/independents/concrete/model-handlers/SyntheticModelHandler.hpp"

#include <iostream>
#include <string>
#include <cassert>

#define make_divisible(v, d) (v + (d - (v % d)))

using namespace std;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;

float SyntheticModelHandler::SetModelField(float *field, vector<float> &model_file,
                                           int nx, int nz, int ny,
                                           int logical_nx, int logical_nz, int logical_ny) {
    float *temp_field = new float[nx * nz * ny];
    memset(temp_field, 0, nx * nz * ny * sizeof(float));
    /// Extracting Velocity and size of each layer in terms of
    /// start(x,y,z) and end(x,y,z)
    float vel;
    int vsx, vsy, vsz, vex, vey, vez;
    float max = 0;
    for (int i = 0; i < model_file.size(); i += 7) {
        vel = model_file[i];
        if (vel > max) {
            max = vel;
        }
        /// Starting filling the velocity with required value
        /// value after shifting the padding of boundaries and half length
        int offset = this->mpParameters->GetBoundaryLength() + this->mpParameters->GetHalfLength();
        vsx = (int) model_file[i + 1] + offset;
        vsz = (int) model_file[i + 2] + offset;
        vsy = (int) model_file[i + 3];
        vex = (int) model_file[i + 4] + offset;
        vez = (int) model_file[i + 5] + offset;
        vey = (int) model_file[i + 6];

        if (ny > 1) {
            vsy = (int) vsy + offset;
            vey = (int) vey + offset;
        }
        if (vsx < 0 || vsy < 0 || vsz < 0) {
            cout << "Error at starting index values (x,y,z) (" << vsx << "," << vsy
                 << "," << vsz << ")" << std::endl;
            continue;
        }
        if (vex > logical_nx || vey > logical_ny || vez > logical_nz) {
            cout << "Error at ending index values (x,y,z) (" << vex << "," << vey
                 << "," << vez << ")" << std::endl;
            continue;
        }

        /// Initialize the layer with the extracted velocity value
#pragma omp parallel default(shared)
        {
#pragma omp for schedule(static) collapse(3)
            for (int y = vsy; y < vey; y++) {
                for (int z = vsz; z < vez; z++) {
                    for (int x = vsx; x < vex; x++) {
                        temp_field[y * nx * nz + z * nx + x] = vel;
                    }
                }
            }
        }
    }
    Device::MemCpy(field, temp_field, nx * nz * ny * sizeof(float), Device::COPY_HOST_TO_DEVICE);
    delete[] temp_field;
    return max;
}

void SyntheticModelHandler::PreprocessModel() {
    int nx = mpGridBox->GetActualGridSize(X_AXIS);
    int ny = mpGridBox->GetActualGridSize(Y_AXIS);
    int nz = mpGridBox->GetActualGridSize(Z_AXIS);

    float dt2 = mpGridBox->GetDT() * mpGridBox->GetDT();

    std::cout << "SyntheticModelHandler::PreprocessModel not implemented" << std::endl;
    assert(0);

}

void SyntheticModelHandler::SetupWindow() {
    if (mpParameters->IsUsingWindow()) {
        uint wnx = mpGridBox->GetActualWindowSize(X_AXIS);
        uint wnz = mpGridBox->GetActualWindowSize(Z_AXIS);
        uint wny = mpGridBox->GetActualWindowSize(Y_AXIS);
        uint nx = mpGridBox->GetActualGridSize(X_AXIS);
        uint nz = mpGridBox->GetActualGridSize(Z_AXIS);
        uint ny = mpGridBox->GetActualGridSize(Y_AXIS);
        uint sx = mpGridBox->GetWindowStart(X_AXIS);
        uint sz = mpGridBox->GetWindowStart(Z_AXIS);
        uint sy = mpGridBox->GetWindowStart(Y_AXIS);
        uint offset = mpParameters->GetHalfLength() + mpParameters->GetBoundaryLength();
        uint start_x = offset;
        uint end_x = mpGridBox->GetLogicalWindowSize(X_AXIS) - offset;
        uint start_z = offset;
        uint end_z = mpGridBox->GetLogicalWindowSize(Z_AXIS) - offset;
        uint start_y = 0;
        uint end_y = 1;
        if (ny != 1) {
            start_y = offset;
            end_y = mpGridBox->GetLogicalWindowSize(Y_AXIS) - offset;
        }
    }
}

void SyntheticModelHandler::SetupPadding() {
    auto grid = mpGridBox;
    auto parameters = mpParameters;
    uint block_x = parameters->GetBlockX();
    uint block_z = parameters->GetBlockZ();
    uint nx = grid->GetLogicalWindowSize(X_AXIS);
    uint nz = grid->GetLogicalWindowSize(Z_AXIS);
    uint inx = nx - 2 * parameters->GetHalfLength();
    uint inz = nz - 2 * parameters->GetHalfLength();
    // Store old values of nx,nz,ny to use in boundaries/etc....

    this->mpGridBox->SetActualGridSize(X_AXIS, this->mpGridBox->GetLogicalGridSize(X_AXIS));
    this->mpGridBox->SetActualGridSize(Y_AXIS, this->mpGridBox->GetLogicalGridSize(Y_AXIS));
    this->mpGridBox->SetActualGridSize(Z_AXIS, this->mpGridBox->GetLogicalGridSize(Z_AXIS));
    this->mpGridBox->SetActualWindowSize(X_AXIS, this->mpGridBox->GetLogicalWindowSize(X_AXIS));
    this->mpGridBox->SetActualWindowSize(Y_AXIS, this->mpGridBox->GetLogicalWindowSize(Y_AXIS));
    this->mpGridBox->SetActualWindowSize(Z_AXIS, this->mpGridBox->GetLogicalWindowSize(Z_AXIS));
    this->mpGridBox->SetComputationGridSize(X_AXIS,
                                            this->mpGridBox->GetLogicalWindowSize(X_AXIS) -
                                            2 * this->mpParameters->GetHalfLength());
    this->mpGridBox->SetComputationGridSize(Z_AXIS,
                                            this->mpGridBox->GetLogicalWindowSize(Z_AXIS) -
                                            2 * this->mpParameters->GetHalfLength());
    if (this->mpGridBox->GetLogicalWindowSize(Y_AXIS) > 1) {
        this->mpGridBox->SetComputationGridSize(Y_AXIS,
                                                this->mpGridBox->GetLogicalWindowSize(Y_AXIS) -
                                                2 * this->mpParameters->GetHalfLength());
    }
    if (block_x > inx) {
        block_x = inx;
        std::cout << "Block Factor x > domain size... Reduced to domain size"
                  << std::endl;
    }
    if (block_z > inz) {
        block_z = inz;
        std::cout << "Block Factor z > domain size... Reduced to domain size"
                  << std::endl;
    }
    if (block_x % 16 != 0 && block_x != 1) {
        block_x = make_divisible(
                block_x,
                16); // Ensure block in x is divisible by 16(biggest vector length).
        std::cout << "Adjusting block factor in x to make it divisible by "
                     "16(Possible Vector Length)..."
                  << std::endl;
    }
    if (inx % block_x != 0) {
        std::cout
                << "Adding padding to make domain divisible by block size in  x-axis"
                << std::endl;
        inx = make_divisible(inx, block_x);
        grid->SetComputationGridSize(X_AXIS, inx);
        nx = inx + 2 * parameters->GetHalfLength();
    }
    if (inz % block_z != 0) {
        std::cout
                << "Adding padding to make domain divisible by block size in  z-axis"
                << std::endl;
        inz = make_divisible(inz, block_z);
        grid->SetComputationGridSize(Z_AXIS, inz);
        nz = inz + 2 * parameters->GetHalfLength();
    }
    if (nx % 16 != 0) {
        std::cout << "Adding padding to ensure alignment of each row" << std::endl;
        nx = make_divisible(nx, 16);
    }
    // Set grid with the padded values.
    grid->SetActualWindowSize(X_AXIS, nx);
    grid->SetActualWindowSize(Z_AXIS, nz);
    parameters->SetBlockX(block_x);
    parameters->SetBlockZ(block_z);
    if (!parameters->IsUsingWindow()) {
        grid->SetActualGridSize(X_AXIS, grid->GetActualWindowSize(X_AXIS));
        grid->SetActualGridSize(Z_AXIS, grid->GetActualWindowSize(Z_AXIS));
    }
}
