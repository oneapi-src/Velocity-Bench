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

#include "operations/components/independents/concrete/trace-writers/BinaryTraceWriter.hpp"

#include <iostream>

using namespace std;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;

BinaryTraceWriter::BinaryTraceWriter(operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mpOutStream = nullptr;
}

BinaryTraceWriter::~BinaryTraceWriter() {
    delete this->mpOutStream;
}

void BinaryTraceWriter::AcquireConfiguration() {}

void BinaryTraceWriter::SetComputationParameters(
        ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void BinaryTraceWriter::SetGridBox(GridBox *apGridBox) {
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

Point3D LocalizePoint(
        Point3D point, bool is_2D, uint half_length, uint bound_length) {
    Point3D copy;
    copy.x = point.x - half_length - bound_length;
    copy.z = point.z - half_length - bound_length;
    if (!is_2D) {
        copy.y = point.y - half_length - bound_length;
    } else {
        copy.y = point.y;
    }
    return copy;
}

void BinaryTraceWriter::InitializeWriter(
        ModellingConfiguration *apModellingConfiguration,
        std::string output_filename) {
    this->mpOutStream = new ofstream(output_filename, ios::out | ios::binary);
    bool is_2D = this->mpGridBox->GetLogicalGridSize(Y_AXIS) == 1;
    Point3D local_source =
            LocalizePoint(apModellingConfiguration->SourcePoint, is_2D,
                          mpParameters->GetHalfLength(), mpParameters->GetBoundaryLength());
    Point3D local_start =
            LocalizePoint(apModellingConfiguration->ReceiversStart, is_2D,
                          mpParameters->GetHalfLength(), mpParameters->GetBoundaryLength());
    Point3D local_end =
            LocalizePoint(apModellingConfiguration->ReceiversEnd, is_2D,
                          mpParameters->GetHalfLength(), mpParameters->GetBoundaryLength());
    this->mpOutStream->write((char *) &local_source, sizeof(local_source));
    this->mpOutStream->write((char *) &local_start, sizeof(local_start));
    this->mpOutStream->write((char *) &apModellingConfiguration->ReceiversIncrement,
                             sizeof(apModellingConfiguration->ReceiversIncrement));
    this->mpOutStream->write((char *) &local_end, sizeof(local_end));
    this->mpOutStream->write((char *) &apModellingConfiguration->TotalTime,
                             sizeof(apModellingConfiguration->TotalTime));
    float dt = this->mpGridBox->GetDT();
    this->mpOutStream->write((char *) &dt, sizeof(float));
    this->mReceiverStart = apModellingConfiguration->ReceiversStart;
    this->mReceiverEnd = apModellingConfiguration->ReceiversEnd;
    this->mReceiverIncrement = apModellingConfiguration->ReceiversIncrement;
}
