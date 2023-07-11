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
// Created by amr on 03/01/2021.
//
#include <operations/backend/OneAPIBackend.hpp>

using namespace operations::backend;

OneAPIBackend::OneAPIBackend() {
    this->mDeviceQueue = nullptr;
    this->mOneAPIAlgorithm = SYCL_ALGORITHM::CPU;
}

OneAPIBackend::~OneAPIBackend() {
    delete this->mDeviceQueue;
}

void OneAPIBackend::SetDeviceQueue(sycl::queue *aDeviceQueue) {
    this->mDeviceQueue = aDeviceQueue;
}

void OneAPIBackend::SetAlgorithm(SYCL_ALGORITHM aOneAPIAlgorithm) {
    this->mOneAPIAlgorithm = aOneAPIAlgorithm;
}
