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
// Created by zeyad-osama on 02/11/2020.
//

#include <thoth/streams/concrete/writers/BinaryWriter.hpp>

#include <thoth/common/ExitCodes.hpp>
#include <thoth/configurations/interface/MapKeys.h>

#include <iostream>
#include <sys/stat.h>

using namespace thoth::streams;
using namespace thoth::common::exitcodes;


BinaryWriter::BinaryWriter(thoth::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
}

void BinaryWriter::AcquireConfiguration() {
    this->mFilePath = this->mpConfigurationMap->GetValue(
            IO_K_PROPERTIES, IO_K_WRITE_PATH, this->mFilePath);
}

int BinaryWriter::Initialize(std::string &aFilePath) {
    mkdir(this->mFilePath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    this->mOutputStream = std::ofstream(this->mFilePath + aFilePath + this->GetExtension(),
                                        std::ios::out | std::ios::binary);
    if (!this->mOutputStream) {
        std::cout << "Cannot open file!" << std::endl;
    }

    return IO_RC_SUCCESS;
}

int BinaryWriter::Write(thoth::dataunits::Gather *aGather) {
    for (int i = 0; i < aGather->GetNumberTraces(); ++i) {
        auto trace = aGather->GetTrace(i);
        auto n_s = trace->GetNumberOfSamples();
        auto trace_data = trace->GetTraceData();
        this->mOutputStream.write((char *) trace_data, n_s * sizeof(float));
    }
    if (!this->mOutputStream.good()) {
        std::cout << "Error occurred at writing time!" << std::endl;
        return 1;
    }
    return 0;
}

int BinaryWriter::Write(std::vector<dataunits::Gather *> aGathers) {
    if (!this->mOutputStream) {
        std::cout << "Cannot open file!" << std::endl;
        return 1;
    }
    for (auto &e : aGathers) {
        this->Write(e);
    }
    return 0;
}

int BinaryWriter::Finalize() {
    this->mOutputStream.close();
    return 0;
}

std::string BinaryWriter::GetExtension() {
    return IO_K_EXT_BIN;
}
