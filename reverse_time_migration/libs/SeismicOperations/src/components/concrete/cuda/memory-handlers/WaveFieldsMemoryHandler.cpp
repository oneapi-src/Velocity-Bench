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
// Created by zeyad-osama on 26/09/2020.
//

#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>

/////#include <operations/backend/OneAPIBackend.hpp>

#include <cassert>

///////using namespace cl::sycl;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;
/////using namespace operations::backend;


void WaveFieldsMemoryHandler::FirstTouch(float *ptr, GridBox *apGridBox, bool enable_window) {
}
