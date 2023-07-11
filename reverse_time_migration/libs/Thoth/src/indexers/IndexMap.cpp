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

#include <thoth/indexers/IndexMap.hpp>

#include <thoth/common/ExitCodes.hpp>

using namespace thoth::indexers;
using namespace thoth::dataunits;


IndexMap::IndexMap() = default;

IndexMap::~IndexMap() = default;

int IndexMap::Reset() {
    this->mIndexMap.clear();
    return IO_RC_SUCCESS;
}

std::map<size_t, std::vector<size_t>>
IndexMap::Get(const TraceHeaderKey &aTraceHeaderKey, std::vector<size_t> &aTraceHeaderValues) {
    std::map<size_t, std::vector<size_t>> bytes;
    for (auto &it : aTraceHeaderValues) {
        bytes[it] = this->Get(aTraceHeaderKey, it);
    }
    return bytes;
}

int IndexMap::Add(const TraceHeaderKey &aTraceHeaderKey,
                  const size_t &aTraceHeaderValue,
                  const size_t &aBytePosition) {
    this->mIndexMap[aTraceHeaderKey][aTraceHeaderValue].push_back(aBytePosition);
    return IO_RC_SUCCESS;
}

int IndexMap::Add(const TraceHeaderKey &aTraceHeaderKey,
                  const size_t &aTraceHeaderValue,
                  const std::vector<size_t> &aBytePositions) {
    int rc = 0;
    for (auto const &it : aBytePositions) {
        rc += this->Add(aTraceHeaderKey, aTraceHeaderValue, aBytePositions);
    }
    /* Check that all Add() functions returned IO_RC_SUCCESS signal. */
    return aBytePositions.empty() ? IO_RC_SUCCESS : (rc / aBytePositions.size()) == IO_RC_SUCCESS;
}
