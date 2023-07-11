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
// Created by mennatallah on 7/9/20.
//

#ifndef PIPELINE_AGENTS_DYNAMIC_SERVER_HPP
#define PIPELINE_AGENTS_DYNAMIC_SERVER_HPP

#if defined(USING_MPI)

#include <stbx/agents/interface/Agent.hpp>

#include <mpi.h>

namespace stbx {
    namespace agents {

        class DynamicServerAgent : public Agent {
        public:
            DynamicServerAgent() = default;

            ~DynamicServerAgent() override;

            operations::dataunits::GridBox *Initialize() override;

            void BeforeMigration() override;

            void AfterMigration() override;

            void BeforeFinalize() override;

            operations::dataunits::MigrationData *AfterFinalize(
                    operations::dataunits::MigrationData *apMigrationData) override;

            bool HasNextShot() override;

            std::vector<uint> GetNextShot() override;

        private:
            uint mCount;

            /// Current process rank is stored in self
            int self, mProcessCount;


            /// the MPI communicator
            MPI_Comm mCommunication;

            /// Reception operation status
            MPI_Status mStatus;

            /// All shots vector
            std::vector<uint> mPossibleShots;

            /// Shots per each process
            std::vector<uint> mProcessShots;

            /// Original shot size
            uint mShotsSize;

            /// Current shot ID to be processed
            int mShotTracker = 0;

            /// Indicate the first shots iterators
            int mShotsCount = 0;

            operations::dataunits::GridBox *mpGridBox;

            std::vector<float *> mStacks;

            /**
             * ==============================================
             * Flags used for the master slave communication.
             * ==============================================
             *
             * flag[0] -> Shot ID (from server to clients).
             *
             * flag[1] -> Shot flag (from server to clients):
             *     - 0 = There are shots available
             *     - 1 = All shots are migrated
             *
             * flag[2] -> Availability flag (from clients to server):
             *     - 0 = Still working
             *     - 1 = Available now
             *
             * flag[3] -> Rank flag (from clients to server)
             *
             * flag[4] -> Migration flag (from server to clients):
             *     - 0 = For a process if (process rank < # of shots) -> It will be working on a shot
             *     - 1 = If (process rank >= Number of shots) -> There is no enough shots for this process
             */
            int flag[5] = {0, 0, 0, 0, 0};
        };
    }//namespace agents
}//namespace stbx

#endif

#endif //PIPELINE_AGENTS_DYNAMIC_SERVER_HPP
