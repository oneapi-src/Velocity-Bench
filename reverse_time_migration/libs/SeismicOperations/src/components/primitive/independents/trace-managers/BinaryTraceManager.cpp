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

#include <operations/components/independents/concrete/trace-managers/BinaryTraceManager.hpp>

#include <operations/utils/interpolation/Interpolator.hpp>

#include <memory-manager/MemoryManager.h>

#include <iostream>
#include <cmath>

using namespace std;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;
using namespace operations::utils::interpolation;


BinaryTraceManager::BinaryTraceManager(operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mAbsoluteShotNumber = 0;
    this->mpTracesHolder = new TracesHolder();
    this->mpTracesHolder->Traces = nullptr;
    this->mInterpolation = NONE;
    this->mTotalTime = 0.0f;
}

BinaryTraceManager::~BinaryTraceManager() {
    if (this->mpTracesHolder->Traces != nullptr) {
        mem_free(this->mpTracesHolder->Traces);
    }
    delete this->mpTracesHolder;
}

void BinaryTraceManager::AcquireConfiguration() {
    if (this->mpConfigurationMap->Contains(OP_K_INTERPOLATION, OP_K_TYPE)) {
        string interpolation = OP_K_NONE;
        interpolation = this->mpConfigurationMap->GetValue(OP_K_INTERPOLATION, OP_K_TYPE, interpolation);
        if (interpolation == OP_K_NONE) {
            this->mInterpolation = NONE;
        } else if (interpolation == OP_K_SPLINE) {
            this->mInterpolation = SPLINE;
        }
    } else {
        cout << "Invalid value for trace-manager->interpolation key : "
                "supported values [ none | spline ]" << std::endl;
        cout << "Using default trace-manager->interpolation value: none..." << std::endl;
    }
}

Point3D BinaryTraceManager::DeLocalizePoint(Point3D point, bool is_2D,
                                            uint half_length,
                                            uint bound_length) {
    Point3D copy;
    copy.x = point.x + half_length + bound_length;
    copy.z = point.z + half_length + bound_length;
    if (!is_2D) {
        copy.y = point.y + half_length + bound_length;
    } else {
        copy.y = point.y;
    }
    return copy;
}

void BinaryTraceManager::ReadShot(vector<string> filenames, uint shot_number, string sort_key) {
    if (this->mpTracesHolder->Traces != nullptr) {
        mem_free(this->mpTracesHolder->Traces);
        this->mpTracesHolder->Traces = nullptr;
    }
    if (shot_number >= filenames.size()) {
        std::cout << "Invalid shot number given : "
                  << shot_number << std::endl;
        exit(EXIT_FAILURE);
    }

    ifstream *trace_file = new ifstream(filenames[shot_number], ios::out | ios::binary);
    if (!trace_file->is_open()) {
        std::cerr << "Couldn't open trace file '"
                  << filenames[shot_number]
                  << "'..." << std::endl;
        exit(EXIT_FAILURE);
    }

    this->mAbsoluteShotNumber++;
    trace_file->read((char *) &this->mSourcePoint, sizeof(this->mSourcePoint));
    trace_file->read((char *) &this->mReceiverStart, sizeof(this->mReceiverStart));
    trace_file->read((char *) &this->mReceiverIncrement, sizeof(this->mReceiverIncrement));
    trace_file->read((char *) &this->mReceiverEnd, sizeof(this->mReceiverEnd));
    trace_file->read((char *) &mTotalTime, sizeof(mTotalTime));
    trace_file->read((char *) &this->mpTracesHolder->SampleDT, sizeof(this->mpTracesHolder->SampleDT));
    // If window model, need to setup the starting point of the window.
    // Handle 3 cases : no room for left window, no room for right window, room for both.
    // Those 3 cases can apply to y-direction as well if 3D.
    if (this->mpParameters->IsUsingWindow()) {
        this->mpGridBox->SetWindowStart(X_AXIS, 0);
        // No room for left window.
        if (this->mSourcePoint.x < this->mpParameters->GetLeftWindow() ||
            (this->mpParameters->GetLeftWindow() == 0 && this->mpParameters->GetRightWindow() == 0)) {
            this->mpGridBox->SetWindowStart(X_AXIS, 0);
            // No room for right window.
        } else if (this->mSourcePoint.x >= this->mpGridBox->GetLogicalGridSize(X_AXIS) -
                                           this->mpParameters->GetBoundaryLength() -
                                           this->mpParameters->GetHalfLength() -
                                           this->mpParameters->GetRightWindow()) {
            this->mpGridBox->SetWindowStart(X_AXIS, this->mpGridBox->GetLogicalGridSize(X_AXIS) -
                                                    this->mpParameters->GetBoundaryLength() -
                                                    this->mpParameters->GetHalfLength() -
                                                    this->mpParameters->GetRightWindow() -
                                                    this->mpParameters->GetLeftWindow() - 1);
            this->mSourcePoint.x = this->mpGridBox->GetLogicalWindowSize(X_AXIS) -
                                   this->mpParameters->GetBoundaryLength() -
                                   this->mpParameters->GetHalfLength() -
                                   this->mpParameters->GetRightWindow() - 1;
        } else {
            this->mpGridBox->SetWindowStart(X_AXIS, this->mSourcePoint.x - this->mpParameters->GetLeftWindow());
            this->mSourcePoint.x = this->mpParameters->GetLeftWindow();
        }
        this->mpGridBox->SetWindowStart(Y_AXIS, 0);
        if (this->mpGridBox->GetLogicalGridSize(Y_AXIS) != 1) {
            if (this->mSourcePoint.y < this->mpParameters->GetBackWindow() ||
                (this->mpParameters->GetFrontWindow() == 0 && this->mpParameters->GetBackWindow() == 0)) {
                this->mpGridBox->SetWindowStart(Y_AXIS, 0);
            } else if (this->mSourcePoint.y >= this->mpGridBox->GetLogicalGridSize(Y_AXIS) -
                                               this->mpParameters->GetBoundaryLength() -
                                               this->mpParameters->GetHalfLength() -
                                               this->mpParameters->GetFrontWindow()) {
                this->mpGridBox->SetWindowStart(Y_AXIS, this->mpGridBox->GetLogicalGridSize(Y_AXIS) -
                                                        this->mpParameters->GetBoundaryLength() -
                                                        this->mpParameters->GetHalfLength() -
                                                        this->mpParameters->GetFrontWindow() -
                                                        this->mpParameters->GetBackWindow() - 1);
                this->mSourcePoint.y = this->mpGridBox->GetWindowStart(Y_AXIS) -
                                       this->mpParameters->GetBoundaryLength() -
                                       this->mpParameters->GetHalfLength() -
                                       this->mpParameters->GetFrontWindow() - 1;
            } else {
                this->mpGridBox->SetWindowStart(Y_AXIS, this->mSourcePoint.y -
                                                        this->mpParameters->GetBackWindow());
                this->mSourcePoint.y = this->mpParameters->GetBackWindow();
            }
        }
    }
    uint num_elements_per_time_step = 0;
    uint num_rec_x = 0;
    uint num_rec_y = 0;
    uint x_inc = this->mReceiverIncrement.x == 0 ? 1 : this->mReceiverIncrement.x;
    uint y_inc = this->mReceiverIncrement.y == 0 ? 1 : this->mReceiverIncrement.y;
    uint z_inc = this->mReceiverIncrement.z == 0 ? 1 : this->mReceiverIncrement.z;

    uint intern_x = this->mpGridBox->GetLogicalWindowSize(X_AXIS) -
                    2 * this->mpParameters->GetHalfLength() -
                    2 * this->mpParameters->GetBoundaryLength();
    uint intern_y = this->mpGridBox->GetLogicalWindowSize(Y_AXIS) -
                    2 * this->mpParameters->GetHalfLength() -
                    2 * this->mpParameters->GetBoundaryLength();

    for (int iz = this->mReceiverStart.z; iz < this->mReceiverEnd.z; iz += z_inc) {
        num_rec_y = 0;
        for (int iy = this->mReceiverStart.y; iy < this->mReceiverEnd.y; iy += y_inc) {
            if (iy >= this->mpGridBox->GetWindowStart(Y_AXIS) &&
                iy < this->mpGridBox->GetWindowStart(Y_AXIS) + intern_y) {
                num_rec_y++;
                num_rec_x = 0;
                for (int ix = this->mReceiverStart.x; ix < this->mReceiverEnd.x; ix += x_inc) {
                    if (ix >= this->mpGridBox->GetWindowStart(X_AXIS) &&
                        ix < this->mpGridBox->GetWindowStart(X_AXIS) + intern_x) {
                        num_elements_per_time_step++;
                        num_rec_x++;
                    }
                }
            }
        }
    }
    this->mpTracesHolder->TraceSizePerTimeStep = num_elements_per_time_step;
    this->mpTracesHolder->ReceiversCountX = num_rec_x;
    this->mpTracesHolder->ReceiversCountY = num_rec_y;
    int sample_nt = int(mTotalTime / this->mpTracesHolder->SampleDT) - 1;
    this->mpTracesHolder->SampleNT = sample_nt;
    this->mpTracesHolder->Traces = (float *) mem_allocate(
            sizeof(float), sample_nt * num_elements_per_time_step, "traces");
    for (int t = 0; t < sample_nt; t++) {
        int index = 0;
        for (int iz = this->mReceiverStart.z; iz < this->mReceiverEnd.z; iz += z_inc) {
            for (int iy = this->mReceiverStart.y; iy < this->mReceiverEnd.y; iy += y_inc) {
                for (int ix = this->mReceiverStart.x; ix < this->mReceiverEnd.x; ix += x_inc) {
                    float value = 0;
                    trace_file->read((char *) &value, sizeof(value));
                    if (iy >= this->mpGridBox->GetWindowStart(Y_AXIS) &&
                        iy < this->mpGridBox->GetWindowStart(Y_AXIS) + intern_y) {
                        if (ix >= this->mpGridBox->GetWindowStart(X_AXIS) &&
                            ix < this->mpGridBox->GetWindowStart(X_AXIS) + intern_x) {
                            this->mpTracesHolder->Traces[t * num_elements_per_time_step + index] = value;
                            index++;
                        }
                    }
                }
            }
        }
    }
    while (this->mReceiverStart.y < this->mpGridBox->GetWindowStart(Y_AXIS)) {
        this->mReceiverStart.y += y_inc;
    }
    while (this->mReceiverEnd.y >= this->mpGridBox->GetWindowStart(Y_AXIS) + intern_y) {
        this->mReceiverEnd.y -= y_inc;
    }
    while (this->mReceiverStart.x < this->mpGridBox->GetWindowStart(X_AXIS)) {
        this->mReceiverStart.x += x_inc;
    }
    while (this->mReceiverEnd.x >= this->mpGridBox->GetWindowStart(X_AXIS) + intern_x) {
        this->mReceiverEnd.x -= x_inc;
    }

    this->mpGridBox->SetNT(int(mTotalTime / this->mpGridBox->GetDT()));

    this->mReceiverStart.x -= this->mpGridBox->GetWindowStart(X_AXIS);
    this->mReceiverStart.y -= this->mpGridBox->GetWindowStart(Y_AXIS);
    this->mReceiverStart.z -= this->mpGridBox->GetWindowStart(Z_AXIS);

    this->mReceiverEnd.x -= this->mpGridBox->GetWindowStart(X_AXIS);
    this->mReceiverEnd.y -= this->mpGridBox->GetWindowStart(Y_AXIS);
    this->mReceiverEnd.z -= this->mpGridBox->GetWindowStart(Z_AXIS);

    delete trace_file;
}

void BinaryTraceManager::PreprocessShot(uint cut_off_time_step) {
    Interpolator::Interpolate(this->mpTracesHolder,
                              this->mpGridBox->GetNT(),
                              this->mTotalTime,
                              this->mInterpolation);

    bool is_2D = this->mpGridBox->GetLogicalGridSize(Y_AXIS) == 1;
    uint half_length = this->mpParameters->GetHalfLength();
    uint bound_length = this->mpParameters->GetBoundaryLength();
    this->mSourcePoint =
            DeLocalizePoint(this->mSourcePoint, is_2D, half_length, bound_length);
    this->mReceiverStart =
            DeLocalizePoint(this->mReceiverStart, is_2D, half_length, bound_length);
    this->mReceiverEnd = DeLocalizePoint(this->mReceiverEnd, is_2D, half_length, bound_length);

    int x_inc = this->mReceiverIncrement.x == 0 ? 1 : this->mReceiverIncrement.x;
    int y_inc = this->mReceiverIncrement.y == 0 ? 1 : this->mReceiverIncrement.y;
    int trace_size = this->mpTracesHolder->TraceSizePerTimeStep;
    int wnx = this->mpGridBox->GetLogicalWindowSize(X_AXIS);
    int wnz_wnx = this->mpGridBox->GetLogicalWindowSize(Z_AXIS) * wnx;
    mpDTraces.Allocate(this->mpTracesHolder->SampleNT * this->mpTracesHolder->TraceSizePerTimeStep,
                       "Device traces");
    Device::MemCpy(mpDTraces.GetNativePointer(), this->mpTracesHolder->Traces,
                   this->mpTracesHolder->SampleNT * this->mpTracesHolder->TraceSizePerTimeStep * sizeof(float),
                   Device::COPY_HOST_TO_DEVICE);
    /*
        // Muting for 2D, not implemented for 3D.
        if (is_2D) {
            float *travel_times = new float[wnx];
            float dt = grid->dt;
            float dx = grid->cell_dimensions.dx;
            for (int i = 0; i < wnx; i++) {
                // Recalculate velocity from the precomputed c2 * dt2
                float velocity = grid->velocity[i + (half_length + bound_length) *
       wnx
                                                +
       grid->window_size.window_start.x]; velocity = velocity / (dt * dt);
                velocity = sqrt(velocity);
                travel_times[i] = dx / velocity;
            }

            for (int i = this->source_point.x + 1; i < wnx - half_length; i++) {
                travel_times[i] = travel_times[i] + travel_times[i - 1];
            }

            for (int i = this->source_point.x - 1; i >= half_length; i--) {
                travel_times[i] = travel_times[i] + travel_times[i + 1];
            }
            int index = 0;
            int sample_nt = int(total_time / traces->sample_dt) - 1;
            for (int iy = r_start.y; iy < r_end.y; iy += y_inc) {
                for (int ix = r_start.x; ix < r_end.x; ix += x_inc) {
                    uint num_times = cut_off_timestep + uint(travel_times[ix] /
       traces->sample_dt); if (num_times > sample_nt) { num_times = sample_nt;
                    }
                    for (uint it = 0; it < num_times; it++) {
                        traces->traces[it * trace_size + index] = 0;
                    }
                    index++;
                }
            }
            delete[] travel_times;
        }
    */
}

void BinaryTraceManager::ApplyIsotropicField() {
    /// @todo To be implemented.
}

void BinaryTraceManager::RevertIsotropicField() {
    /// @todo To be implemented.
}

TracesHolder *BinaryTraceManager::GetTracesHolder() { return this->mpTracesHolder; }

void BinaryTraceManager::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void BinaryTraceManager::SetGridBox(GridBox *apGridBox) {
    this->mpGridBox = apGridBox;
    if (this->mpGridBox == nullptr) {
        std::cerr << "No GridBox provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    /* Does not support 3D. */
    if (this->mpGridBox->GetActualWindowSize(Y_AXIS) > 1) {
        throw exceptions::NotImplementedException();
    }
}

Point3D *BinaryTraceManager::GetSourcePoint() {
    return &this->mSourcePoint;
}

vector<uint>
BinaryTraceManager::GetWorkingShots(vector<string> filenames, uint min_shot, uint max_shot, string type) {
    vector<uint> all_shots;
    uint end_val = max_shot < filenames.size() - 1 ? max_shot : filenames.size() - 1;;
    for (uint i = min_shot; i <= end_val; i++) {
        all_shots.push_back(i);
    }
    return all_shots;
}
