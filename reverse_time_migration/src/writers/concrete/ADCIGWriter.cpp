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
// Created by zeyad-osama on 09/09/2020.
//

#include <stbx/writers/concrete/ADCIGWriter.hpp>

#include <cmath>

using namespace std;
using namespace stbx::writers;
using namespace operations::utils::filters;
using namespace operations::utils::io;


ADCIGWriter::ADCIGWriter() {
    this->mRawMigration = nullptr;
    this->mRawMigrationIntervals = nullptr;
    this->mRawMigrationStacked = nullptr;

    this->mFilteredMigration = nullptr;
    this->mFilteredMigrationIntervals = nullptr;
    this->mFilteredMigrationStacked = nullptr;
}

ADCIGWriter::~ADCIGWriter() {
    delete[] this->mRawMigrationIntervals;
    delete[] this->mRawMigrationStacked;

    delete[] this->mFilteredMigration;
    delete[] this->mFilteredMigrationIntervals;
    delete[] this->mFilteredMigrationStacked;
}

void ADCIGWriter::Initialize() {
    uint stacked_size = this->mpMigrationData->GetGridSize(X_AXIS) *
                        this->mpMigrationData->GetGridSize(Y_AXIS) *
                        this->mpMigrationData->GetGridSize(Z_AXIS);

    uint cig_size = stacked_size * this->mpMigrationData->GetGatherDimension();

    uint intervals_size = cig_size / this->mIntervalLength;

    this->mFilteredMigration = new float[cig_size];

    this->mRawMigrationIntervals = new float[intervals_size];
    this->mFilteredMigrationIntervals = new float[intervals_size];

    this->mRawMigrationStacked = new float[stacked_size];
    this->mFilteredMigrationStacked = new float[stacked_size];
    this->mRawMigration = this->mpMigrationData->GetResultAt(0)->GetData();
}

void ADCIGWriter::Filter() {
    uint offset = this->mpMigrationData->GetGridSize(X_AXIS) *
                  this->mpMigrationData->GetGridSize(Y_AXIS) *
                  this->mpMigrationData->GetGridSize(Z_AXIS);

    for (uint theta = 0; theta < mpMigrationData->GetGatherDimension(); theta++) {
        float *raw_plane = this->mRawMigration + theta * offset;
        float *filtered_plane = this->mFilteredMigration + theta * offset;
        apply_laplace_filter(raw_plane,
                             filtered_plane,
                             this->mpMigrationData->GetGridSize(X_AXIS),
                             this->mpMigrationData->GetGridSize(Y_AXIS),
                             this->mpMigrationData->GetGridSize(Z_AXIS));
    }
}

void ADCIGWriter::PrepareResults() {
    uint nx = this->mpMigrationData->GetGridSize(X_AXIS);
    uint ny = this->mpMigrationData->GetGridSize(Y_AXIS);
    uint nz = this->mpMigrationData->GetGridSize(Z_AXIS);
    uint n_angles = this->mpMigrationData->GetGatherDimension();

    /// Creating stacked images
    for (uint iy = 0; iy < ny; iy++) {
        for (uint iz = 0; iz < nz; iz++) {
            for (uint ix = 0; ix < nx; ix++) {
                uint offset = iy * nz * nx + iz * nx + ix;

                float value_r = 0;
                float value_f = 0;

                for (int theta = 0; theta < n_angles; theta++) {
                    value_r += this->mRawMigration[theta * ny * nz * nx + offset];
                    value_f += this->mFilteredMigration[theta * ny * nz * nx + offset];
                }

                this->mRawMigrationStacked[offset] = value_r;
                this->mFilteredMigrationStacked[offset] = value_f;
            }
        }
    }

    // creating interval images
    int modified_nx = (nx / this->mIntervalLength) * n_angles;

    for (int iy = 0; iy < ny; iy++) {
        for (int iz = 0; iz < nz; iz++) {
            for (int ix = 0; ix < nx; ix++) {

                for (uint theta = 0; theta < n_angles; theta++) {

                    uint id = theta * nx * nz * ny + iy * nz * nx + iz * nx + ix;

                    if (ix % this->mIntervalLength == 0) {
                        uint id_o = iy * nz * modified_nx + iz * modified_nx +
                                    (ix / this->mIntervalLength) * n_angles +
                                    theta;

                        this->mRawMigrationIntervals[id_o] =
                                this->mRawMigration[id];

                        this->mFilteredMigrationIntervals[id_o] =
                                this->mFilteredMigration[id];
                    }
                }
            }
        }
    }
}

void ADCIGWriter::WriteSegyIntervals(float *frame, const string &file_name) {
    string file_name_extension = file_name + ".segy";

    uint modified_nx =
            std::floor(this->mpMigrationData->GetGridSize(X_AXIS) /
                       float(this->mIntervalLength)) *
            this->mpMigrationData->GetGatherDimension();

    write_segy(modified_nx,
               this->mpMigrationData->GetGridSize(Y_AXIS),
               this->mpMigrationData->GetGridSize(Z_AXIS),
               this->mpMigrationData->GetNT(),
               this->mpMigrationData->GetCellDimensions(X_AXIS),
               this->mpMigrationData->GetCellDimensions(Y_AXIS),
               this->mpMigrationData->GetCellDimensions(Z_AXIS),
               this->mpMigrationData->GetDT(),
               frame, file_name_extension, false);
}

void ADCIGWriter::WriteCIG(float *frame, const string &file_name) {
    string file_name_extension = file_name + ".segy";

    write_adcig_segy(this->mpMigrationData->GetGridSize(X_AXIS),
                     this->mpMigrationData->GetGridSize(Y_AXIS),
                     this->mpMigrationData->GetGridSize(Z_AXIS),
                     this->mpMigrationData->GetNT(),
                     this->mpMigrationData->GetGatherDimension(),
                     this->mpMigrationData->GetCellDimensions(X_AXIS),
                     this->mpMigrationData->GetCellDimensions(Y_AXIS),
                     this->mpMigrationData->GetCellDimensions(Z_AXIS),
                     this->mpMigrationData->GetDT(),
                     frame, file_name_extension, false);
}

void ADCIGWriter::Write(const string &write_path, bool is_traces) {
    this->Initialize();
    this->SpecifyRawMigration();
    this->PostProcess();
    this->Filter();
    this->PrepareResults();

    WriteSegy(this->mRawMigrationStacked, write_path + "/raw_migration_stacked");
    WriteSegy(this->mFilteredMigrationStacked, write_path + "/filtered_migration_stacked");

    WriteSegyIntervals(this->mRawMigrationIntervals, write_path + "/raw_migration_intervals");
    WriteSegyIntervals(this->mFilteredMigrationIntervals, write_path + "/filtered_migration_intervals");

//    WriteCIG(this->mRawMigration, write_path + "/raw_migration");

    WriteBinary(this->mRawMigrationStacked, write_path + "/raw_migration");
    WriteBinary(this->mFilteredMigrationStacked, write_path + "/filtered_migration");

#ifdef ENABLE_GPU_TIMINGS
    WriteTimeResults(write_path);
#endif
}
