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
// Created by zeyad-osama on 26/09/2020.
//

#include "operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp"

#include <memory-manager/MemoryManager.h>

#include <cmath>

using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;

WaveFieldsMemoryHandler::WaveFieldsMemoryHandler(
        operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
}

WaveFieldsMemoryHandler::~WaveFieldsMemoryHandler() = default;

void WaveFieldsMemoryHandler::AcquireConfiguration() {}

void WaveFieldsMemoryHandler::CloneWaveFields(GridBox *_src, GridBox *_dst) {
    uint wnx = _src->GetActualWindowSize(X_AXIS);
    uint wny = _src->GetActualWindowSize(Y_AXIS);
    uint wnz = _src->GetActualWindowSize(Z_AXIS);
    uint const window_size = wnx * wny * wnz;

    /// Allocating and zeroing wave fields.
    for (auto const &wave_field : _src->GetWaveFields()) {
        if (!GridBox::Includes(wave_field.first, NEXT)) {
            auto frame_buffer = new FrameBuffer<float>();

            frame_buffer->Allocate(window_size,
                                   mpParameters->GetHalfLength(),
                                   GridBox::Stringify(wave_field.first));

            this->FirstTouch(frame_buffer->GetNativePointer(), _src, true);
            _dst->RegisterWaveField(wave_field.first, frame_buffer);


            Device::MemCpy(_dst->Get(wave_field.first)->GetNativePointer(),
                           _src->Get(wave_field.first)->GetNativePointer(),
                           window_size * sizeof(float), Device::COPY_DEVICE_TO_DEVICE);

            GridBox::Key wave_field_new = wave_field.first;

            if (GridBox::Includes(wave_field_new, GB_PRSS | PREV)) {
                GridBox::Replace(&wave_field_new, PREV, NEXT);
                _dst->RegisterWaveField(wave_field_new, frame_buffer);
            } else if (this->mpParameters->GetEquationOrder() == FIRST &&
                       GridBox::Includes(wave_field_new, GB_PRSS | CURR)) {
                GridBox::Replace(&wave_field_new, CURR, NEXT);
                _dst->RegisterWaveField(wave_field_new, frame_buffer);
            }
        }
    }
}

void WaveFieldsMemoryHandler::CopyWaveFields(GridBox *_src, GridBox *_dst) {
    uint wnx = _src->GetActualWindowSize(X_AXIS);
    uint wny = _src->GetActualWindowSize(Y_AXIS);
    uint wnz = _src->GetActualWindowSize(Z_AXIS);
    uint const window_size = wnx * wny * wnz;

    for (auto const &wave_field : _src->GetWaveFields()) {
        auto src = _src->Get(wave_field.first)->GetNativePointer();
        auto dst = _dst->Get(wave_field.first)->GetNativePointer();

        if (src != nullptr && dst != nullptr) {
            Device::MemCpy(dst, src, window_size * sizeof(float), Device::COPY_DEVICE_TO_DEVICE);
        } else {
            std::cerr << "No Wave Fields allocated to be copied... "
                      << "Terminating..." << std::endl;
            exit(EXIT_FAILURE);
        }

    }
}

void WaveFieldsMemoryHandler::FreeWaveFields(GridBox *apGridBox) {
    for (auto &wave_field : apGridBox->GetWaveFields()) {
        if (!GridBox::Includes(wave_field.first, NEXT)) {
            wave_field.second->Free();
        }
    }
}

void WaveFieldsMemoryHandler::SetComputationParameters(
        ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}
