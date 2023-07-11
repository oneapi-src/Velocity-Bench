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
// Created by zeyad-osama on 11/03/2021.
//

#include <thoth/indexers/FileIndexer.hpp>

#include <thoth/lookups/mappers/TraceHeaderMapper.hpp>
#include <thoth/utils/convertors/NumbersConvertor.hpp>
#include <thoth/configurations/interface/MapKeys.h>

#include <iostream>

#define IO_INDEX_NAME   "_index"

using namespace thoth::indexers;
using namespace thoth::dataunits;
using namespace thoth::lookups;
using namespace thoth::streams::helpers;
using namespace thoth::configuration;
using namespace thoth::utils::convertors;

/// @todo To be removed
/// {
#include <thoth/exceptions/Exceptions.hpp>

using namespace thoth::exceptions;
/// }


FileIndexer::FileIndexer(std::string &aFilePath)
        : mInFilePath(aFilePath), mOutFilePath(GenerateOutFilePathName(aFilePath)) {
    this->mInStreamHelper = nullptr;
    this->mOutStreamHelper = nullptr;
}

FileIndexer::~FileIndexer() {
    delete this->mInStreamHelper;
    delete this->mOutStreamHelper;
}

size_t FileIndexer::Initialize() {
    this->mInStreamHelper = new InStreamHelper(this->mInFilePath);
    this->mOutStreamHelper = new OutStreamHelper(this->mOutFilePath);
    return (this->mInStreamHelper->Open() && this->mOutStreamHelper->Open());
}

int FileIndexer::Finalize() {
    return (this->mInStreamHelper->Close() && this->mOutStreamHelper->Close());
}

std::string FileIndexer::GenerateOutFilePathName(std::string &aInFileName) {
    return aInFileName.substr(0, aInFileName.size() - std::string(IO_K_EXT_SGY).length())
           + IO_INDEX_NAME
           + IO_K_EXT_SGY_INDEX;
}

IndexMap FileIndexer::Index(const std::vector<TraceHeaderKey> &aTraceHeaderKeys) {
    /* Read binary header in the given file.*/
    auto bhl = this->mInStreamHelper->ReadBinaryHeader(IO_POS_S_BINARY_HEADER);
    unsigned long long file_size = this->mInStreamHelper->GetFileSize();
    size_t start_pos = IO_POS_S_TRACE_HEADER;
    while (true) {
        if (start_pos + IO_SIZE_TRACE_HEADER >= file_size) {
            break;
        }
        /* Read trace header in the given file. */
        auto thl = this->mInStreamHelper->ReadTraceHeader(start_pos);

        /// @todo
//        /* Loop upon trace header keys to generate the corresponding index map from. */
//        for (const auto it : aTraceHeaderKeys) {
//            this->mIndexMap.Add(it, TraceHeaderMapper::GetTraceHeaderValue(it, thl), start_pos);
//        }
        ///@todo To be removed
        /// {
        this->mIndexMap.Add(TraceHeaderKey::FLDR, NumbersConvertor::ToLittleEndian(thl.FLDR), start_pos);
        /// {

        /* Update stream position pointer. */
        start_pos += IO_SIZE_TRACE_HEADER + InStreamHelper::GetTraceDataSize(thl, bhl);
    }


    ///@todo To be removed
    /// {
//    for (const auto &k : aTraceHeaderKeys) {
//        for (const auto &map : this->mIndexMap.Get(TraceHeaderKey(k))) {
//            std::cout << "VAL -> " << map.first << std::endl;
//            for (auto &it : map.second) {
//                std::cout << it << " - ";
//            }
//            std::cout << std::endl;
//        }
//    }
    /// }
    return this->mIndexMap;
}

int FileIndexer::Flush() {
    /// @todo To be removed
    /// {
    throw NotImplementedException();
    /// }
}
