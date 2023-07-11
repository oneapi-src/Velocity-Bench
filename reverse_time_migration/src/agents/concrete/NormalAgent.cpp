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
// Created by zeyad-osama on 01/09/2020.
//

#include <stbx/agents/concrete/NormalAgent.hpp>

using namespace std;
using namespace stbx::agents;
using namespace operations::dataunits;

NormalAgent::~NormalAgent() = default;

GridBox *NormalAgent::Initialize() {
    return Agent::Initialize();
}

void NormalAgent::BeforeMigration() {}

void NormalAgent::AfterMigration() {}

void NormalAgent::BeforeFinalize() {}

MigrationData *NormalAgent::AfterFinalize(MigrationData *aMigrationData) {
    return aMigrationData;
}

bool NormalAgent::HasNextShot() {
    this->mCount++;
    return this->mCount < 2;
}

vector<uint> NormalAgent::GetNextShot() {
    return this->mpEngine->GetValidShots();
}
