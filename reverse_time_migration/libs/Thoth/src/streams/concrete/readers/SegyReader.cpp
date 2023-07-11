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

#include <thoth/streams/concrete/readers/SegyReader.hpp>

#include <thoth/streams/helpers/InStreamHelper.hpp>
#include <thoth/data-units/helpers/TraceHelper.hpp>
#include <thoth/lookups/tables/TextHeaderLookup.hpp>
#include <thoth/lookups/tables/BinaryHeaderLookup.hpp>
#include <thoth/lookups/tables/TraceHeaderLookup.hpp>
#include <thoth/utils/convertors/NumbersConvertor.hpp>
#include <thoth/utils/convertors/StringsConvertor.hpp>
#include <thoth/utils/convertors/FloatingPointFormatter.hpp>
#include <thoth/common/ExitCodes.hpp>

#define IO_K_FIRST_OCCURRENCE   0   /* First occurrence position */

/// @todo To be removed
/// {
#include <thoth/exceptions/Exceptions.hpp>

using namespace thoth::exceptions;
/// }

using namespace thoth::streams;
using namespace thoth::streams::helpers;
using namespace thoth::lookups;
using namespace thoth::indexers;
using namespace thoth::dataunits;
using namespace thoth::dataunits::helpers;
using namespace thoth::common::exitcodes;
using namespace thoth::utils::convertors;


SegyReader::SegyReader(thoth::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mEnableHeaderOnly = false;
    this->mHasExtendedTextHeader = false;
    this->mStoreTextHeaders = false;
    this->mTextHeader = nullptr;
    this->mExtendedHeader = nullptr;
}

SegyReader::~SegyReader() {
    delete this->mTextHeader;
    if (this->mHasExtendedTextHeader) {
        delete this->mExtendedHeader;
    }
}

void SegyReader::AcquireConfiguration() {
    this->mEnableHeaderOnly = this->mpConfigurationMap->GetValue(
            IO_K_PROPERTIES, IO_K_TEXT_HEADERS_ONLY, this->mEnableHeaderOnly);
    this->mStoreTextHeaders = this->mpConfigurationMap->GetValue(
            IO_K_PROPERTIES, IO_K_TEXT_HEADERS_STORE, this->mStoreTextHeaders);
}

std::string SegyReader::GetExtension() {
    return IO_K_EXT_SGY;
}

int SegyReader::Initialize(std::vector<std::string> &aGatherKeys,
                           std::vector<std::pair<std::string, Gather::SortDirection>> &aSortingKeys,
                           std::vector<std::string> &aPaths) {
    /// @todo To be removed
    /// {
    throw NotImplementedException();
    /// }
}

int SegyReader::Initialize(std::vector<TraceHeaderKey> &aGatherKeys,
                           std::vector<std::pair<TraceHeaderKey, dataunits::Gather::SortDirection>> &aSortingKeys,
                           std::vector<std::string> &aPaths) {
    /* Reset All Variables to start from scratch */
    this->mInStreamHelpers.clear();
    this->mFileIndexers.clear();
    this->mIndexMaps.clear();
    this->mEnableHeaderOnly = false;
    this->mHasExtendedTextHeader = false;
    this->mStoreTextHeaders = false;
    this->mTextHeader = nullptr;
    this->mExtendedHeader = nullptr;
    /* Internal variables initializations. */
    this->mGatherKeys = aGatherKeys;
    this->mSortingKeys = aSortingKeys;
    this->mPaths = aPaths;

    /* Initialize streams. */
    for (auto &it : this->mPaths) {
        this->mInStreamHelpers.push_back(new InStreamHelper(it));
    }
    /* Open streams. */
    for (auto &it : this->mInStreamHelpers) {
        it->Open();
    }

    /* Read text header in the given file. */
    if (this->mStoreTextHeaders) {
        this->mTextHeader = this->mInStreamHelpers[IO_K_FIRST_OCCURRENCE]->ReadTextHeader(IO_POS_S_TEXT_HEADER);
    }

    /* Read binary header in the given file and check whether an
     * extended text header is available or not for later reading */
    this->mBinaryHeaderLookup = this->mInStreamHelpers[IO_K_FIRST_OCCURRENCE]->ReadBinaryHeader(IO_POS_S_BINARY_HEADER);
    this->mHasExtendedTextHeader = NumbersConvertor::ToLittleEndian(this->mBinaryHeaderLookup.EXT_HEAD);

    /* Read extended text header in the given file if found. */
    if (this->mHasExtendedTextHeader && this->mStoreTextHeaders) {
        this->mExtendedHeader = this->mInStreamHelpers[IO_K_FIRST_OCCURRENCE]->ReadTextHeader(IO_POS_S_EXT_TEXT_HEADER);
    }
    /* Index passed files. */
    this->Index();

    return IO_RC_SUCCESS;
}

int SegyReader::Finalize() {
    int rc = 0;
    /* Close streams. */
    for (auto &it : this->mInStreamHelpers) {
        rc += it->Close();
    }
    /* Check that all Write() functions returned IO_RC_SUCCESS signal. */
    return (rc / this->mInStreamHelpers.size()) == IO_RC_SUCCESS;
}

void SegyReader::SetHeaderOnlyMode(bool aEnableHeaderOnly) {
    this->mEnableHeaderOnly = aEnableHeaderOnly;
}

std::vector<Gather *> SegyReader::ReadAll() {
    std::unordered_map<int, std::vector<dataunits::Trace *>> gather_map;
    auto format = NumbersConvertor::ToLittleEndian(this->mBinaryHeaderLookup.FORMAT);

    for (const auto &it : this->mInStreamHelpers) {
        unsigned long long file_size = it->GetFileSize();
        size_t start_pos = IO_POS_S_TRACE_HEADER;
        while (true) {
            if (start_pos + IO_SIZE_TRACE_HEADER >= file_size) {
                break;
            }
            /* Read trace header in the given file. */
            auto thl = it->ReadTraceHeader(start_pos);

            /* Read trace data in the given file. */
            auto trace = it->ReadFormattedTraceData(start_pos + IO_SIZE_TRACE_HEADER, thl, this->mBinaryHeaderLookup);

            gather_map[NumbersConvertor::ToLittleEndian(thl.FLDR)].push_back(trace);

            /* Update stream position pointer. */
            start_pos += IO_SIZE_TRACE_HEADER +
                         FloatingPointFormatter::GetFloatArrayRealSize(NumbersConvertor::ToLittleEndian(thl.NS),
                                                                       format);
        }
    }
    std::vector<Gather *> gathers;
    int16_t hdt = NumbersConvertor::ToLittleEndian(this->mBinaryHeaderLookup.HDT);
    for (auto const &traces : gather_map) {
        auto g = new Gather();
        g->AddTrace(traces.second);
        g->SetSamplingRate(hdt);
        gathers.push_back(g);
    }
    return gathers;
}

Gather *SegyReader::Read(std::vector<std::string> aHeaderValues) {
    if (aHeaderValues.size() != this->mGatherKeys.size()) {
        return nullptr;
    }

    std::unordered_map<int, std::vector<dataunits::Trace *>> gather_map;
    auto gather = new Gather();
    gather->SetSamplingRate(NumbersConvertor::ToLittleEndian(this->mBinaryHeaderLookup.HDT));
    for (int ig = 0; ig < this->mIndexMaps.size(); ++ig) {
        for (int ik = 0; ik < this->mGatherKeys.size(); ++ik) {
            auto bytes = this->mIndexMaps[ig].Get(this->mGatherKeys[ik],
                                                  StringsConvertor::ToLong(aHeaderValues[ik]));
            if (!bytes.empty()) {
                auto stream = this->mInStreamHelpers[ig];
                for (auto &pos : bytes) {
                    /* Read trace header in the given file. */
                    auto thl = stream->ReadTraceHeader(pos);
                    /* Read trace data in the given file. */
                    auto trace = stream->ReadFormattedTraceData(pos + IO_SIZE_TRACE_HEADER, thl,
                                                                this->mBinaryHeaderLookup);
                    gather->AddTrace(trace);
                }
            }
        }
    }
    return gather;
}

std::vector<Gather *> SegyReader::Read(std::vector<std::vector<std::string>> aHeaderValues) {
    /// @todo To be removed
    /// {
    throw NotImplementedException();
    /// }
}

Gather *SegyReader::Read(unsigned int aIndex) {
    /// @todo To be removed
    /// {
    throw NotImplementedException();
    /// }
}

std::vector<std::vector<std::string>> SegyReader::GetIdentifiers() {
    std::vector<std::vector<std::string>> keys;
    for (auto &mIndexMap : this->mIndexMaps) {
        auto map = mIndexMap.Get();
        for (auto &key : this->mGatherKeys) {
            for (const auto &entry : map[key]) {
                if (!entry.second.empty()) {
                    std::vector<std::string> val = {std::to_string(entry.first)};
                    keys.push_back(val);
                }
            }
        }
    }
    return keys;
}

unsigned int SegyReader::GetNumberOfGathers() {
    unsigned int gather_number = 0;
    for (auto &mIndexMap : this->mIndexMaps) {
        auto map = mIndexMap.Get();
        for (const auto &entry : map[this->mGatherKeys[0]]) {
            if (!entry.second.empty()) {
                gather_number++;
            }
        }
    }
    return gather_number;
}

bool SegyReader::HasExtendedTextHeader() const {
    return this->mHasExtendedTextHeader;
}

unsigned char *SegyReader::GetTextHeader() {
    unsigned char *text_header = nullptr;
    if (this->mTextHeader != nullptr) {
        text_header = StringsConvertor::E2A(this->mTextHeader, IO_SIZE_TEXT_HEADER);
    }
    return text_header;
}

unsigned char *SegyReader::GetExtendedTextHeader() {
    unsigned char *text_header = nullptr;
    if (this->mTextHeader != nullptr) {
        text_header = StringsConvertor::E2A(this->mExtendedHeader, IO_SIZE_TEXT_HEADER);
    }
    return text_header;
}

int SegyReader::Index() {
    this->mFileIndexers.reserve(this->mPaths.size());
    for (auto &it : this->mPaths) {
        this->mFileIndexers.push_back(FileIndexer(it));
    }
    for (auto &it : this->mFileIndexers) {
        it.Initialize();
        this->mIndexMaps.push_back(it.Index(this->mGatherKeys));
    }
    return IO_RC_SUCCESS;
}
