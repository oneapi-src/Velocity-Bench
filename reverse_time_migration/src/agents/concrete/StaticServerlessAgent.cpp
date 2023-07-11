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

#if defined(USING_MPI)

#include <stbx/agents/concrete/StaticServerlessAgent.hpp>

#include <mpi.h>

using namespace std;
using namespace stbx::agents;
using namespace operations::dataunits;

StaticServerlessAgent::StaticServerlessAgent() {
    this->mpGridBox = nullptr;
    this->self = 0;
    this->mProcessCount = 0;
    this->mCommunication = 0;
    this->mCount = 0;
}

StaticServerlessAgent::~StaticServerlessAgent() {
    for (auto it : this->mStacks) {
        delete[] it;
    }
}

GridBox *StaticServerlessAgent::Initialize() {
    this->mpGridBox = mpEngine->Initialize();

    int provided;
    MPI_Init_thread(&this->argc, &this->argv, MPI_THREAD_FUNNELED, &provided);

    if (provided != MPI_THREAD_FUNNELED) {
        std::cerr << "Warning MPI did not provide MPI_THREAD_FUNNELED..." << std::endl;
    }

    this->mCommunication = MPI_COMM_WORLD;
    MPI_Comm_rank(this->mCommunication, &this->self);
    MPI_Comm_size(this->mCommunication, &this->mProcessCount);

    return this->mpGridBox;
}

void StaticServerlessAgent::BeforeMigration() {
    this->mPossibleShots = mpEngine->GetValidShots();
}

void StaticServerlessAgent::AfterMigration() {}

void StaticServerlessAgent::BeforeFinalize() {}

MigrationData *StaticServerlessAgent::AfterFinalize(MigrationData *apMigrationData) {
    MigrationData *md = apMigrationData;

    uint size = md->GetGridSize(X_AXIS) *
                md->GetGridSize(Y_AXIS) *
                md->GetGridSize(Z_AXIS) *
                md->GetGatherDimension();

    for (int i = 0; i < md->GetResults().size(); ++i) {
        this->mStacks.push_back(new float[size]);
        MPI_Reduce(md->GetResultAt(i)->GetData(), this->mStacks[i],
                   size, MPI_FLOAT, MPI_SUM, 0, this->mCommunication);
    }

    if (this->self == 0) {
        for (int i = 0; i < md->GetResults().size(); ++i) {
            md->SetResults(i, new Result(this->mStacks[i]));
        }
    }

    MPI_Finalize();
    if (this->self != 0) {
        exit(0);
    }

    return md;
}

bool StaticServerlessAgent::HasNextShot() {
    this->mCount++;
    return this->mCount < 2;
}

vector<uint> StaticServerlessAgent::GetNextShot() {
    vector<uint> process_shots;
    uint size = this->mPossibleShots.size();
    for (int i = this->self; i < size; i = i + this->mProcessCount) {
        process_shots.push_back(this->mPossibleShots[i]);
    }
    return process_shots;
}

#endif
