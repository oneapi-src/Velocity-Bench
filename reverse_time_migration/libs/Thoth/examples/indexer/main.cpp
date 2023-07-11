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
// Created by zeyad-osama on 07/03/2021.
//

#include <thoth/utils/timer/ExecutionTimer.hpp>
#include <thoth/indexers/FileIndexer.hpp>
#include <thoth/indexers/IndexMap.hpp>

#include <iostream>

using namespace std;
using namespace thoth::indexers;
using namespace thoth::dataunits;
using namespace thoth::utils::timer;


int main(int argc, char *argv[]) {
    vector<TraceHeaderKey> vec = {TraceHeaderKey::FLDR};
    string file_path = DATA_PATH "/shots0601_0800.segy";
    FileIndexer fi(file_path);
    fi.Initialize();

    ExecutionTimer::Evaluate([&]() {
        fi.Index(vec);
    }, true);
}
