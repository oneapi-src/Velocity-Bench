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
// Created by ingy-mounir on 1/28/20.
//

#include "operations/components/independents/concrete/trace-managers/SeismicTraceManager.hpp"

#include <timer/Timer.h>
#include <memory-manager/MemoryManager.h>
#include <operations/utils/interpolation/Interpolator.hpp>
#include <operations/utils/io/read_utils.h>

#include <iostream>
#include <utility>
#include <cmath>

using namespace std;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;
using namespace operations::utils::interpolation;
using namespace thoth::streams;
using namespace thoth::dataunits;

SeismicTraceManager::SeismicTraceManager(
        operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mInterpolation = NONE;
    this->mpTracesHolder = new TracesHolder();
    nlohmann::json configuration_map;
    configuration_map[IO_K_PROPERTIES][IO_K_TEXT_HEADERS_ONLY] = false;
    configuration_map[IO_K_PROPERTIES][IO_K_TEXT_HEADERS_STORE] = false;
    thoth::configuration::JSONConfigurationMap io_conf_map(
            configuration_map);
    this->mpSeismicReader = new SegyReader(&io_conf_map);
    this->mpSeismicReader->AcquireConfiguration();
    this->mTotalTime = 0.0;
    this->mShotStride = 1;
}

SeismicTraceManager::~SeismicTraceManager() {
    if (this->mpTracesHolder->Traces != nullptr) {
        mem_free(this->mpTracesHolder->Traces);
        mem_free(this->mpTracesHolder->PositionsX);
        mem_free(this->mpTracesHolder->PositionsY);
    }
    delete this->mpSeismicReader;
    delete this->mpTracesHolder;
}

void SeismicTraceManager::AcquireConfiguration() {
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

    this->mShotStride = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_SHOT_STRIDE, this->mShotStride);
    std::cout << "Using Shot Stride = " << this->mShotStride << " for trace-manager" << std::endl;
}

void SeismicTraceManager::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void SeismicTraceManager::SetGridBox(GridBox *apGridBox) {
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

Point3D SeismicTraceManager::SDeLocalizePoint(Point3D point, bool is_2D,
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

IPoint3D SeismicTraceManager::DeLocalizePointS(IPoint3D aIPoint3D,
                                               bool is_2D,
                                               uint half_length,
                                               uint bound_length) {
    IPoint3D copy;
    copy.x = aIPoint3D.x + half_length + bound_length;
    copy.z = aIPoint3D.z + half_length + bound_length;
    if (!is_2D) {
        copy.y = aIPoint3D.y + half_length + bound_length;
    } else {
        copy.y = aIPoint3D.y;
    }
    return copy;
}

void SeismicTraceManager::ReadShot(vector<string> file_names,
                                   uint shot_number,
                                   string sort_key) {
    if (this->mpTracesHolder->Traces != nullptr) {
        mem_free(this->mpTracesHolder->Traces);
        mem_free(this->mpTracesHolder->PositionsX);
        mem_free(this->mpTracesHolder->PositionsY);

        this->mpTracesHolder->Traces = nullptr;
        this->mpTracesHolder->PositionsX = nullptr;
        this->mpTracesHolder->PositionsY = nullptr;
    }
    Gather *gather;
#ifdef ENABLE_GPU_TIMINGS
    Timer *timer = Timer::GetInstance();
    timer->StartTimer("IO::ReadSelectedShotFromSegyFile");
#endif
    gather = this->mpSeismicReader->Read({std::to_string(shot_number)});
#ifdef ENABLE_GPU_TIMINGS
    timer->StopTimer("IO::ReadSelectedShotFromSegyFile");
#endif

    if (gather == nullptr) {
        std::cerr << "Didn't find a suitable file to read shot ID "
                  << shot_number
                  << " from..." << std::endl;
        exit(EXIT_FAILURE);
    } else {
        std::cout << "Reading trace for shot ID "
                  << shot_number
                  << std::endl;
    }

    utils::io::ParseGatherToTraces(gather,
                                   &this->mpSourcePoint,
                                   this->mpTracesHolder,
                                   &this->mpTracesHolder->PositionsX,
                                   &this->mpTracesHolder->PositionsY,
                                   this->mpGridBox,
                                   this->mpParameters,
                                   &this->mTotalTime);
    delete gather;
}

void SeismicTraceManager::PreprocessShot(uint cut_off_time_step) {
    Interpolator::Interpolate(this->mpTracesHolder,
                              this->mpGridBox->GetNT(),
                              this->mTotalTime,
                              this->mInterpolation);

    bool is_2D = this->mpGridBox->GetLogicalGridSize(Y_AXIS) == 1;
    uint half_length = mpParameters->GetHalfLength();
    uint bound_length = mpParameters->GetBoundaryLength();
    this->mpSourcePoint =
            SDeLocalizePoint(this->mpSourcePoint, is_2D, half_length, bound_length);
    mpDTraces.Allocate(this->mpTracesHolder->SampleNT * this->mpTracesHolder->TraceSizePerTimeStep,
                       "Device traces");
    mpDPositionsX.Allocate(this->mpTracesHolder->TraceSizePerTimeStep, "trace x positions");
    mpDPositionsY.Allocate(this->mpTracesHolder->TraceSizePerTimeStep, "trace y positions");
    Device::MemCpy(mpDTraces.GetNativePointer(), this->mpTracesHolder->Traces,
                   this->mpTracesHolder->SampleNT * this->mpTracesHolder->TraceSizePerTimeStep * sizeof(float),
                   Device::COPY_HOST_TO_DEVICE);
    Device::MemCpy(mpDPositionsX.GetNativePointer(), this->mpTracesHolder->PositionsX,
                   this->mpTracesHolder->TraceSizePerTimeStep * sizeof(uint),
                   Device::COPY_HOST_TO_DEVICE);
    Device::MemCpy(mpDPositionsY.GetNativePointer(), this->mpTracesHolder->PositionsY,
                   this->mpTracesHolder->TraceSizePerTimeStep * sizeof(uint),
                   Device::COPY_HOST_TO_DEVICE);
}

void SeismicTraceManager::ApplyIsotropicField() {
    /// @todo To be implemented.
}

void SeismicTraceManager::RevertIsotropicField() {
    /// @todo To be implemented.
}

TracesHolder *SeismicTraceManager::GetTracesHolder() {
    return this->mpTracesHolder;
}

Point3D *SeismicTraceManager::GetSourcePoint() {
    return &this->mpSourcePoint;
}

vector<uint> SeismicTraceManager::GetWorkingShots(
        vector<string> file_names, uint min_shot, uint max_shot, string type) {
    std::vector<TraceHeaderKey> gather_keys = {TraceHeaderKey::FLDR};
    std::vector<std::pair<TraceHeaderKey, Gather::SortDirection>> sorting_keys;
    this->mpSeismicReader->Initialize(gather_keys, sorting_keys, file_names);
    auto keys = this->mpSeismicReader->GetIdentifiers();
    vector<uint> all_shots;
    for (auto const &key : keys) {
        size_t unique_val = std::stoull(key[0]);
        if (unique_val >= min_shot && unique_val <= max_shot) {
            all_shots.push_back(unique_val);
        }
    }

    vector<uint> selected_shots;
    for (int i = 0; i < all_shots.size(); i += this->mShotStride) {
        selected_shots.push_back(all_shots[i]);
    }

    return selected_shots;
}
