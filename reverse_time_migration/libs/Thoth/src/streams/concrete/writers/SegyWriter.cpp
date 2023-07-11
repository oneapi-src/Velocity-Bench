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

#include <thoth/streams/concrete/writers/SegyWriter.hpp>

#include <thoth/streams/helpers/OutStreamHelper.hpp>
#include <thoth/common/ExitCodes.hpp>

using namespace thoth::streams;
using namespace thoth::streams::helpers;
using namespace thoth::common::exitcodes;


SegyWriter::SegyWriter(thoth::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mOutStreamHelpers = nullptr;
    this->mWriteLittleEndian = true;
}

SegyWriter::~SegyWriter() {
    delete this->mOutStreamHelpers;
}

void SegyWriter::AcquireConfiguration() {
    this->mWriteLittleEndian = this->mpConfigurationMap->GetValue(
            IO_K_PROPERTIES, IO_K_WRITE_LITTLE_ENDIAN, this->mWriteLittleEndian);
}

std::string SegyWriter::GetExtension() {
    return IO_K_EXT_SGY;
}

int SegyWriter::Initialize(std::string &aFilePath) {
    this->mFilePath = aFilePath;
    this->mOutStreamHelpers = new OutStreamHelper(this->mFilePath);
    this->mOutStreamHelpers->Open();
    return IO_RC_SUCCESS;
}

int SegyWriter::Finalize() {
    this->mOutStreamHelpers->Close();
    return IO_RC_SUCCESS;
}

int SegyWriter::Write(std::vector<dataunits::Gather *> aGathers) {
    int rc = 0;
    for (const auto &it : aGathers) {
        rc += this->Write(it);
    }
    /* Check that all Write() functions returned IO_RC_SUCCESS signal. */
    return aGathers.empty() ? IO_RC_SUCCESS : (rc / aGathers.size()) == IO_RC_SUCCESS;
}

int SegyWriter::Write(thoth::dataunits::Gather *aGather) {
    return 0;
}
