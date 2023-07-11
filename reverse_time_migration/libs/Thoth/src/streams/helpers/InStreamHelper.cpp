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
// Created by zeyad-osama on 10/03/2021.
//

#include <thoth/streams/helpers/InStreamHelper.hpp>

#include <thoth/exceptions/Exceptions.hpp>
#include <thoth/utils/convertors/NumbersConvertor.hpp>
#include <thoth/utils/convertors/FloatingPointFormatter.hpp>
#include <thoth/data-units/helpers/TraceHelper.hpp>
#include <thoth/common/ExitCodes.hpp>

#include <iostream>

using namespace thoth::streams::helpers;
using namespace thoth::dataunits;
using namespace thoth::dataunits::helpers;
using namespace thoth::lookups;
using namespace thoth::exceptions;
using namespace thoth::utils::convertors;
using namespace thoth::common::exitcodes;


InStreamHelper::InStreamHelper(std::string &aFilePath)
        : mFilePath(aFilePath), mFileSize(-1) {}

InStreamHelper::~InStreamHelper() = default;

size_t InStreamHelper::Open() {
    this->mInStream.open(this->mFilePath.c_str(), std::ifstream::in);
    if (this->mInStream.fail()) {
        std::cerr << "Error opening file  " << this->mFilePath << std::endl;
        exit(EXIT_FAILURE);
    }
    return this->GetFileSize();
}

int InStreamHelper::Close() {
    this->mInStream.close();
    return IO_RC_SUCCESS;
}

unsigned char *InStreamHelper::ReadBytesBlock(size_t aStartPosition, size_t aBlockSize) {
    if (aStartPosition + aBlockSize > this->GetFileSize()) {
        throw IndexOutOfBoundsException();
    }
    auto buffer = new unsigned char[aBlockSize];
    memset(buffer, '\0', sizeof(unsigned char) * aBlockSize);
    this->mInStream.seekg(aStartPosition, std::fstream::beg);
    this->mInStream.read((char *) buffer, aBlockSize);
    return buffer;
}

unsigned char *InStreamHelper::ReadTextHeader(size_t aStartPosition) {
    if (aStartPosition + IO_SIZE_TEXT_HEADER > this->GetFileSize()) {
        throw IndexOutOfBoundsException();
    }
    return this->ReadBytesBlock(aStartPosition, IO_SIZE_TEXT_HEADER);
}

BinaryHeaderLookup InStreamHelper::ReadBinaryHeader(size_t aStartPosition) {
    if (aStartPosition + IO_SIZE_BINARY_HEADER > this->GetFileSize()) {
        throw IndexOutOfBoundsException();
    }
    auto bhl_buffer = this->ReadBytesBlock(aStartPosition, IO_SIZE_BINARY_HEADER);
    BinaryHeaderLookup bhl{};
    std::memcpy(&bhl, bhl_buffer, sizeof(BinaryHeaderLookup));
    delete [] bhl_buffer;
    return bhl;
}

TraceHeaderLookup InStreamHelper::ReadTraceHeader(size_t aStartPosition) {
    if (aStartPosition + IO_SIZE_TRACE_HEADER > this->GetFileSize()) {
        throw IndexOutOfBoundsException();
    }
    auto thl_buffer = this->ReadBytesBlock(aStartPosition, IO_SIZE_TRACE_HEADER);
    TraceHeaderLookup thl{};
    std::memcpy(&thl, thl_buffer, sizeof(TraceHeaderLookup));
    delete [] thl_buffer;
    return thl;
}

Trace *InStreamHelper::ReadFormattedTraceData(size_t aStartPosition,
                                              const TraceHeaderLookup &aTraceHeaderLookup,
                                              const BinaryHeaderLookup &aBinaryHeaderLookup) {
    auto trace_size = InStreamHelper::GetTraceDataSize(aTraceHeaderLookup, aBinaryHeaderLookup);

    if (aStartPosition + trace_size > this->GetFileSize()) {
        throw IndexOutOfBoundsException();
    }

    auto trace_data = new char[trace_size];
    this->mInStream.seekg(aStartPosition, std::fstream::beg);
    this->mInStream.read(trace_data, trace_size);

    FloatingPointFormatter::Format(trace_data, trace_size,
                                   InStreamHelper::GetSamplesNumber(aTraceHeaderLookup, aBinaryHeaderLookup),
                                   NumbersConvertor::ToLittleEndian(aBinaryHeaderLookup.FORMAT));

    auto trace = new Trace(NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.NS));
    trace->SetTraceData((float *) trace_data);
    /* Set of trace IDs. */
    trace->SetTraceHeaderKeyValue(TraceHeaderKey(TraceHeaderKey::FLDR),
                                  NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.FLDR));
    trace->SetTraceHeaderKeyValue(TraceHeaderKey(TraceHeaderKey::TRACL),
                                  NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.TRACL));
    trace->SetTraceHeaderKeyValue(TraceHeaderKey(TraceHeaderKey::TRACF),
                                  NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.TRACF));
    trace->SetTraceHeaderKeyValue(TraceHeaderKey(TraceHeaderKey::TRACR),
                                  NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.TRACR));

    /* Source locations. */
    trace->SetTraceHeaderKeyValue(TraceHeaderKey(TraceHeaderKey::SX),
                                  NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.SX));
    trace->SetTraceHeaderKeyValue(TraceHeaderKey(TraceHeaderKey::SY),
                                  NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.SY));
    /* Receivers locations. */
    trace->SetTraceHeaderKeyValue(TraceHeaderKey(TraceHeaderKey::GX),
                                  NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.GX));
    trace->SetTraceHeaderKeyValue(TraceHeaderKey(TraceHeaderKey::GY),
                                  NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.GY));

    /* Scaling Factors. */
    trace->SetTraceHeaderKeyValue(TraceHeaderKey::SCALCO, NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.SCALCO));
    trace->SetTraceHeaderKeyValue(TraceHeaderKey::SCALEL, NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.SCALEL));

    /* Weight trace data values according to the target formats. */
    TraceHelper::Weight(trace, aTraceHeaderLookup, aBinaryHeaderLookup);

    return trace;
}

size_t InStreamHelper::GetFileSize() {
    if (this->mFileSize == -1) {
        size_t curr_offset = this->mInStream.tellg();
        this->mInStream.seekg(0, std::fstream::end);
        this->mFileSize = this->mInStream.tellg();
        this->mInStream.seekg(curr_offset, std::fstream::beg);

    }
    return this->mFileSize;
}

size_t InStreamHelper::GetCurrentPosition() {
    return this->mInStream.tellg();
}

size_t InStreamHelper::GetTraceDataSize(const TraceHeaderLookup &aTraceHeaderLookup,
                                        const BinaryHeaderLookup &aBinaryHeaderLookup) {
    auto format = NumbersConvertor::ToLittleEndian(aBinaryHeaderLookup.FORMAT);
    auto samples_number = InStreamHelper::GetSamplesNumber(aTraceHeaderLookup, aBinaryHeaderLookup);
    return FloatingPointFormatter::GetFloatArrayRealSize(samples_number, format);
}

size_t InStreamHelper::GetSamplesNumber(const TraceHeaderLookup &aTraceHeaderLookup,
                                        const BinaryHeaderLookup &aBinaryHeaderLookup) {
    auto ns = NumbersConvertor::ToLittleEndian(aTraceHeaderLookup.NS);
    auto hns = NumbersConvertor::ToLittleEndian(aBinaryHeaderLookup.HNS);

    auto samples_number = ns;
    if (samples_number == 0) {
        samples_number = hns;
    }
    return samples_number;
}
