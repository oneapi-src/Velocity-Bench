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

#include <thoth/streams/concrete/writers/SUWriter.hpp>

#include <thoth/common/ExitCodes.hpp>
#include <thoth/configurations/interface/MapKeys.h>
#include <thoth/helpers/stream_helpers.h>

#include <iostream>
#include <sys/stat.h>
#include <cstring>

using namespace thoth::streams;
using namespace thoth::streams::helpers;
using namespace thoth::dataunits;
using namespace thoth::common::exitcodes;


SUWriter::SUWriter(thoth::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mWriteLittleEndian = true;
}

void SUWriter::AcquireConfiguration() {
    this->mWriteLittleEndian = this->mpConfigurationMap->GetValue(
            IO_K_PROPERTIES, IO_K_WRITE_LITTLE_ENDIAN, this->mWriteLittleEndian);
    this->mFilePath = this->mpConfigurationMap->GetValue(
            IO_K_PROPERTIES, IO_K_WRITE_PATH, this->mFilePath);
}

int SUWriter::Initialize(std::string &aFilePath) {
    mkdir(this->mFilePath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    this->mOutputStream = std::ofstream(this->mFilePath + aFilePath + this->GetExtension(),
                                        std::ios::out | std::ios::binary);
    if (!this->mOutputStream) {
        std::cout << "Cannot open file!" << std::endl;
    }

    return IO_RC_SUCCESS;
}

int SUWriter::Write(thoth::dataunits::Gather *aGather) {
    bool swap_bytes = true;
    if ((this->mWriteLittleEndian && is_little_endian_machine()) ||
        (!this->mWriteLittleEndian && !is_little_endian_machine())) {
        swap_bytes = false;
    }
    for (int i = 0; i < aGather->GetNumberTraces(); ++i) {
        auto trace = aGather->GetTrace(i);
        auto n_s = trace->GetNumberOfSamples();
        auto trace_data = trace->GetTraceData();

        // Write trace headers.
        char *trace_headers;
        trace_headers = (char *) malloc(TRACE_HEADERS_BYTES);
        memset(trace_headers, 0, TRACE_HEADERS_BYTES);
        auto fill_trace_headers_return = fill_trace_headers(trace_headers, trace, swap_bytes);
        if (fill_trace_headers_return != 0) {
            free(trace_headers);
            return 1;
        }
        this->mOutputStream.write(trace_headers, TRACE_HEADERS_BYTES);
        free(trace_headers);
        trace_headers = nullptr;

        // Write trace data.
        for (int iz = 0; iz < n_s; ++iz) {
            auto write_value = trace_data[iz];
            if (swap_bytes) {
                write_value = reverse_bytes(&trace_data[iz]);
            }
            this->mOutputStream.write((char *) &write_value, sizeof(float));
        }
    }
    if (!this->mOutputStream.good()) {
        std::cout << "Error occurred at writing time!" << std::endl;
        return 1;
    }
    return 0;
}

int SUWriter::Write(std::vector<dataunits::Gather *> aGathers) {
    if (!this->mOutputStream) {
        std::cout << "Cannot open file!" << std::endl;
        return 1;
    }
    for (auto &e : aGathers) {
        this->Write(e);
    }
    return 0;
}

int SUWriter::Finalize() {
    this->mOutputStream.close();
    return 0;
}

std::string SUWriter::GetExtension() {
    return IO_K_EXT_SU;
}
