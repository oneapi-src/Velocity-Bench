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
// Created by amr-nasr on 19/12/2019.
//

#include <operations/components/independents/concrete/forward-collectors/TwoPropagation.hpp>

#include <operations/utils/compressor/Compressor.hpp>

#include <timer/Timer.h>

#include <sys/stat.h>

using namespace std;
using namespace operations::helpers;
using namespace operations::components;
using namespace operations::components::helpers;
using namespace operations::common;
using namespace operations::dataunits;
using namespace operations::utils::compressors;


TwoPropagation::TwoPropagation(operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mpInternalGridBox = new GridBox();
    this->mpForwardPressure = nullptr;
    this->mIsMemoryFit = false;
    this->mTimeCounter = 0;
    this->mIsCompression = false;
    this->mZFP_Tolerance = 0.01f;
    this->mZFP_Parallel = true;
    this->mZFP_IsRelative = false;
    this->mMaxNT = 0;
    this->mMaxDeviceNT = 0;
    this->mpMaxNTRatio = 0;
}

TwoPropagation::~TwoPropagation() {
    if (this->mpForwardPressureHostMemory != nullptr) {
        mem_free(this->mpForwardPressureHostMemory);
    }
    delete this->mpForwardPressure;
    this->mpWaveFieldsMemoryHandler->FreeWaveFields(this->mpInternalGridBox);
    delete this->mpInternalGridBox;
}

void TwoPropagation::AcquireConfiguration() {
    if (this->mpConfigurationMap->Contains(OP_K_PROPRIETIES, OP_K_COMPRESSION)) {
        this->mIsCompression = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_COMPRESSION,
                                                                  this->mIsCompression);
    }
    if (this->mpConfigurationMap->Contains(OP_K_PROPRIETIES, OP_K_WRITE_PATH)) {
        std::string write_path = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_WRITE_PATH,
                                                                    this->mWritePath);
        mkdir(write_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        this->mWritePath = write_path + "/two_prop";
        mkdir(this->mWritePath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    if (this->mpConfigurationMap->Contains(OP_K_PROPRIETIES, OP_K_ZFP_TOLERANCE)) {
        this->mZFP_Tolerance = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_ZFP_TOLERANCE,
                                                                  this->mZFP_Tolerance);
    }
    if (this->mpConfigurationMap->Contains(OP_K_PROPRIETIES, OP_K_ZFP_PARALLEL)) {
        this->mZFP_Parallel = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_ZFP_PARALLEL,
                                                                 this->mZFP_Parallel);
    }
    if (this->mpConfigurationMap->Contains(OP_K_PROPRIETIES, OP_K_ZFP_RELATIVE)) {
        this->mZFP_IsRelative = this->mpConfigurationMap->GetValue(OP_K_PROPRIETIES, OP_K_ZFP_RELATIVE,
                                                                   this->mZFP_IsRelative);
    }
}

void TwoPropagation::FetchForward() {
    uint wnx = this->mpMainGridBox->GetActualWindowSize(X_AXIS);
    uint wny = this->mpMainGridBox->GetActualWindowSize(Y_AXIS);
    uint wnz = this->mpMainGridBox->GetActualWindowSize(Z_AXIS);
    uint const window_size = wnx * wny * wnz;
    // Retrieve data from files to host buffer
    if ((this->mTimeCounter + 1) % this->mMaxNT == 0) {
        if (this->mIsCompression) {
            Timer *timer = Timer::GetInstance();
            timer->StartTimer("ForwardCollector::Decompression");
            string str = this->mWritePath + "/temp_" + to_string(this->mTimeCounter / this->mMaxNT);
            Compressor::Decompress(this->mpForwardPressureHostMemory,
                                   this->mpMainGridBox->GetActualWindowSize(X_AXIS),
                                   this->mpMainGridBox->GetActualWindowSize(Y_AXIS),
                                   this->mpMainGridBox->GetActualWindowSize(Z_AXIS),
                                   this->mMaxNT,
                                   (double) this->mZFP_Tolerance,
                                   this->mZFP_Parallel,
                                   str.c_str(),
                                   this->mZFP_IsRelative);
            timer->StopTimer("ForwardCollector::Decompression");
        } else {
            Timer *timer = Timer::GetInstance();
            timer->StartTimer("IO::ReadForward");
            string str = this->mWritePath + "/temp_" + to_string(this->mTimeCounter / this->mMaxNT);
            bin_file_load(str.c_str(), this->mpForwardPressureHostMemory, this->mMaxNT * window_size);
            timer->StopTimer("IO::ReadForward");
        }
    }
    // Retrieve data from host buffer
    if ((this->mTimeCounter + 1) % this->mMaxDeviceNT == 0) {

        int host_index = (this->mTimeCounter + 1) / this->mMaxDeviceNT - 1;

        Device::MemCpy(this->mpForwardPressure->GetNativePointer(),
                       this->mpForwardPressureHostMemory +
                       (host_index % this->mpMaxNTRatio) * (this->mMaxDeviceNT * window_size),
                       this->mMaxDeviceNT * window_size * sizeof(float),
                       Device::COPY_HOST_TO_DEVICE);
    }
    this->mpInternalGridBox->Set(WAVE | GB_PRSS | CURR | DIR_Z,
                                 this->mpForwardPressure->GetNativePointer() +
                                 ((this->mTimeCounter) % this->mMaxDeviceNT) * window_size);
    this->mTimeCounter--;
}

void TwoPropagation::ResetGrid(bool aIsForwardRun) {
    uint wnx = this->mpMainGridBox->GetActualWindowSize(X_AXIS);
    uint wny = this->mpMainGridBox->GetActualWindowSize(Y_AXIS);
    uint wnz = this->mpMainGridBox->GetActualWindowSize(Z_AXIS);
    uint const window_size = wnx * wny * wnz;

    if (aIsForwardRun) {
        this->mpMainGridBox->CloneMetaData(this->mpInternalGridBox);
        this->mpMainGridBox->CloneParameters(this->mpInternalGridBox);

        if (this->mpInternalGridBox->GetWaveFields().empty()) {
            this->mpWaveFieldsMemoryHandler->CloneWaveFields(this->mpMainGridBox,
                                                             this->mpInternalGridBox);
        } else {
            this->mpWaveFieldsMemoryHandler->CopyWaveFields(this->mpMainGridBox,
                                                            this->mpInternalGridBox);
        }


        this->mTimeCounter = 0;
        if (this->mpForwardPressureHostMemory == nullptr) {
            /// Add one for empty timeframe at the start of the simulation
            /// (The first previous) since SaveForward is called before each step.
            this->mMaxNT = this->mpMainGridBox->GetNT() + 1;


            this->mMaxDeviceNT = 100; // save 100 frames in the Device memory, then reflect to host memory

            this->mpForwardPressureHostMemory = (float *) mem_allocate(
                    (sizeof(float)), this->mMaxNT * window_size, "forward_pressure");

            this->mpForwardPressure = new FrameBuffer<float>();
            this->mpForwardPressure->Allocate(window_size * this->mMaxDeviceNT);

            if (this->mpForwardPressureHostMemory != nullptr) {
                this->mIsMemoryFit = true;
            } else {
                this->mIsMemoryFit = false;
                while (this->mpForwardPressureHostMemory == nullptr) {
                    this->mMaxNT = this->mMaxNT / 2;
                    this->mpForwardPressureHostMemory = (float *) mem_allocate(
                            (sizeof(float)), this->mMaxNT * window_size, "forward_pressure");
                }

                mem_free(this->mpForwardPressureHostMemory);

                // another iteration as a safety measure
                this->mMaxNT = this->mMaxNT / 2;
                this->mpForwardPressureHostMemory = (float *) mem_allocate(
                        (sizeof(float)), this->mMaxNT * window_size, "forward_pressure");
            }

            mpMaxNTRatio = mMaxNT / this->mMaxDeviceNT;

        }

        this->mpTempCurr = this->mpMainGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
        this->mpTempNext = this->mpMainGridBox->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();

        Device::MemSet(this->mpForwardPressure->GetNativePointer(), 0.0f,
                       window_size * sizeof(float));
        Device::MemSet(this->mpForwardPressure->GetNativePointer() + window_size, 0.0f,
                       window_size * sizeof(float));


        if (this->mpParameters->GetEquationOrder() == SECOND) {
            this->mpTempPrev = this->mpMainGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();

            Device::MemSet(this->mpForwardPressure->GetNativePointer() + 2 * window_size, 0.0f,
                           window_size * sizeof(float));
        }


        if (this->mpParameters->GetEquationOrder() == SECOND) {
            this->mpMainGridBox->Set(WAVE | GB_PRSS | PREV | DIR_Z, this->mpForwardPressure->GetNativePointer());
            this->mpMainGridBox->Set(WAVE | GB_PRSS | CURR | DIR_Z,
                                     this->mpForwardPressure->GetNativePointer() + window_size);
            // Save forward is called before the kernel in the engine.
            // When called will advance pressure next to the right point.
            this->mpMainGridBox->Set(WAVE | GB_PRSS | NEXT | DIR_Z,
                                     this->mpForwardPressure->GetNativePointer() + window_size);
        } else {
            this->mpMainGridBox->Set(WAVE | GB_PRSS | CURR | DIR_Z, this->mpForwardPressure->GetNativePointer());
            this->mpMainGridBox->Set(WAVE | GB_PRSS | NEXT | DIR_Z,
                                     this->mpForwardPressure->GetNativePointer() + window_size);
        }
    } else {
        Device::MemSet(this->mpTempCurr, 0.0f, window_size * sizeof(float));
        if (this->mpParameters->GetEquationOrder() == SECOND) {
            Device::MemSet(this->mpTempPrev, 0.0f, window_size * sizeof(float));
        }

        for (auto const &wave_field : this->mpMainGridBox->GetWaveFields()) {
            if (GridBox::Includes(wave_field.first, GB_PRTC)) {
                Device::MemSet(wave_field.second->GetNativePointer(), 0.0f, window_size * sizeof(float));
            }
        }

        if (!this->mIsMemoryFit) {
            this->mTimeCounter++;
            this->mpInternalGridBox->Set(WAVE | GB_PRSS | CURR | DIR_Z,
                                         this->mpMainGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer());
        } else {
            // Pressure size will be minimized in FetchForward call at first step.
            this->mpInternalGridBox->Set(WAVE | GB_PRSS | CURR | DIR_Z,
                                         this->mpMainGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer() +
                                         window_size);
        }

        if (this->mpParameters->GetEquationOrder() == SECOND) {
            this->mpMainGridBox->Set(WAVE | GB_PRSS | PREV | DIR_Z, this->mpTempPrev);
        }
        this->mpMainGridBox->Set(WAVE | GB_PRSS | CURR | DIR_Z, this->mpTempCurr);
        this->mpMainGridBox->Set(WAVE | GB_PRSS | NEXT | DIR_Z, this->mpTempNext);
    }
}

void TwoPropagation::SaveForward() {
    uint wnx = this->mpMainGridBox->GetActualWindowSize(X_AXIS);
    uint wny = this->mpMainGridBox->GetActualWindowSize(Y_AXIS);
    uint wnz = this->mpMainGridBox->GetActualWindowSize(Z_AXIS);
    uint const window_size = wnx * wny * wnz;

    this->mTimeCounter++;

    // Transfer from Device memory to host memory
    if ((this->mTimeCounter + 1) % this->mMaxDeviceNT == 0) {

        int host_index = (this->mTimeCounter + 1) / this->mMaxDeviceNT - 1;

        Device::MemCpy(
                this->mpForwardPressureHostMemory +
                (host_index % this->mpMaxNTRatio) * (this->mMaxDeviceNT * window_size),
                this->mpForwardPressure->GetNativePointer(),
                this->mMaxDeviceNT * window_size * sizeof(float),
                Device::COPY_DEVICE_TO_HOST);
    }

    // Save host memory to file
    if ((this->mTimeCounter + 1) % this->mMaxNT == 0) {
        if (this->mIsCompression) {
            Timer *timer = Timer::GetInstance();
            timer->StartTimer("ForwardCollector::Compression");
            string str = this->mWritePath + "/temp_" + to_string(this->mTimeCounter / this->mMaxNT);
            Compressor::Compress(this->mpForwardPressureHostMemory,
                                 this->mpMainGridBox->GetActualWindowSize(X_AXIS),
                                 this->mpMainGridBox->GetActualWindowSize(Y_AXIS),
                                 this->mpMainGridBox->GetActualWindowSize(Z_AXIS),
                                 this->mMaxNT,
                                 (double) this->mZFP_Tolerance,
                                 this->mZFP_Parallel,
                                 str.c_str(),
                                 this->mZFP_IsRelative);
            timer->StopTimer("ForwardCollector::Compression");
        } else {
            Timer *timer = Timer::GetInstance();
            timer->StartTimer("IO::WriteForward");
            string str =
                    this->mWritePath + "/temp_" + to_string(this->mTimeCounter / this->mMaxNT);
            bin_file_save(str.c_str(), this->mpForwardPressureHostMemory, this->mMaxNT * window_size);
            timer->StopTimer("IO::WriteForward");
        }
    }

    this->mpMainGridBox->Set(WAVE | GB_PRSS | CURR | DIR_Z,
                             this->mpForwardPressure->GetNativePointer() +
                             ((this->mTimeCounter) % this->mMaxDeviceNT) * window_size);
    this->mpMainGridBox->Set(WAVE | GB_PRSS | NEXT | DIR_Z,
                             this->mpForwardPressure->GetNativePointer() +
                             ((this->mTimeCounter + 1) % this->mMaxDeviceNT) * window_size);
    if (this->mpParameters->GetEquationOrder() == SECOND) {
        this->mpMainGridBox->Set(WAVE | GB_PRSS | PREV | DIR_Z,
                                 this->mpForwardPressure->GetNativePointer() +
                                 ((this->mTimeCounter - 1) % this->mMaxDeviceNT) * window_size);
    }
}

void TwoPropagation::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void TwoPropagation::SetGridBox(GridBox *apGridBox) {
    this->mpMainGridBox = apGridBox;
    if (this->mpMainGridBox == nullptr) {
        std::cout << "Not a compatible GridBox... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    /* Does not support 3D. */
    if (this->mpMainGridBox->GetActualWindowSize(Y_AXIS) > 1) {
        throw exceptions::NotImplementedException();
    }

    /*
     * in case of two propagation the next buffer and previous buffer get separated
     * so we need to register a new wave field with actual allocated memory
     */
    auto framebuffer = new FrameBuffer<float>();
    this->mpMainGridBox->RegisterWaveField(WAVE | GB_PRSS | NEXT | DIR_Z, framebuffer);

    framebuffer->Allocate(
            mpMainGridBox->GetActualWindowSize(X_AXIS) *
            mpMainGridBox->GetActualWindowSize(Y_AXIS) *
            mpMainGridBox->GetActualWindowSize(Z_AXIS),
            mpParameters->GetHalfLength(),
            "next pressure");
}

void TwoPropagation::SetDependentComponents(
        ComponentsMap<DependentComponent> *apDependentComponentsMap) {
    HasDependents::SetDependentComponents(apDependentComponentsMap);

    this->mpWaveFieldsMemoryHandler =
            (WaveFieldsMemoryHandler *)
                    this->GetDependentComponentsMap()->Get(MEMORY_HANDLER);
    if (this->mpWaveFieldsMemoryHandler == nullptr) {
        std::cerr << "No Wave Fields Memory Handler provided... "
                  << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

GridBox *TwoPropagation::GetForwardGrid() {
    return this->mpInternalGridBox;
}
