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
// Created by mennatallah on 6/9/20.
//

#if defined(USING_MPI)

#include <stbx/agents/concrete//StaticServerAgent.hpp>

#include <mpi.h>

using namespace std;
using namespace stbx::agents;
using namespace operations::dataunits;

StaticServerAgent::StaticServerAgent() {
    this->mpGridBox = nullptr;
    this->self = 0;
    this->mProcessCount = 0;
    this->mCount = 0;
    this->mCommunication = 0;
}

StaticServerAgent::~StaticServerAgent() {
    for (auto it : this->mStacks) {
        delete[] it;
    }
}

GridBox *StaticServerAgent::Initialize() {
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

void StaticServerAgent::BeforeMigration() {
    this->mPossibleShots = mpEngine->GetValidShots();
}

void StaticServerAgent::AfterMigration() {}

void StaticServerAgent::BeforeFinalize() {}

MigrationData *StaticServerAgent::AfterFinalize(MigrationData *apMigrationData) {
    MigrationData *md = apMigrationData;

    uint size = md->GetGridSize(X_AXIS) *
                md->GetGridSize(Y_AXIS) *
                md->GetGridSize(Z_AXIS) *
                md->GetGatherDimension();

    for (int i = 0; i < md->GetResults().size(); ++i) {
        mStacks.push_back(new float[size]);
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

bool StaticServerAgent::HasNextShot() {
    this->mCount++;
    return this->mCount > 1 ? false : true;
}

vector<uint> StaticServerAgent::GetNextShot() {
    vector<uint> process_shots;
    if (this->self == 0) {
        process_shots.clear();
        return process_shots;
    }

    uint size = mPossibleShots.size();
    for (int i = this->self - 1; i < size; i = i + (this->mProcessCount - 1)) {
        process_shots.push_back(this->mPossibleShots[i]);
    }
    return process_shots;
}

#endif
