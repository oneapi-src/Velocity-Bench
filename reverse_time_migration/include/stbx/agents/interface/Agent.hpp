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

#ifndef PIPELINE_AGENTS_AGENT_HPP
#define PIPELINE_AGENTS_AGENT_HPP

#include <operations/engines/interface/Engine.hpp>

namespace stbx {
    namespace agents {
        /**
         * @brief After specifying which running approach is currently
         * being used (i.e. MPI/No MPI), Agent is responsible to preform
         * RTMEngine tasks accordingly.
         *
         * @note Should be inherited whenever a new approach is used
         */
        class Agent {
        public:
            /**
             * @brief Constructor should be overridden to
             * initialize needed  member variables.
             */
            Agent() = default;

            /**
             * @brief Destructor should be overridden to
             * ensure correct memory management.
             */
            virtual ~Agent() = default;

            /**
             * @brief Assign initialized engine to agent to use in
             * all functions.
             * @param aEngine : RTMEngine
             */
            inline void AssignEngine(operations::engines::Engine *aEngine) {
                mpEngine = aEngine;
            }

            /**
             * @brief Assign CLI arguments to agent to use in
             * all functions.
             * @param _argc
             * @param _argv
             */
            inline void AssignArgs(int _argc, char **_argv) {
                argc = _argc;
                argv = _argv;
            }

            /**
             * @brief Initialize member variables assigned to
             * each derived class.
             * <br>
             * Should always be called in each derived class as follows:-
             *      GridBox *gb = Agent::Initialize();
             * and return gb at the end of each derived Initialize function.
             *
             * @return GridBox*
             */
            virtual operations::dataunits::GridBox *Initialize() {
                return mpEngine->Initialize();
            }

            /**
             * @brief Preform all tasks need by the engine before migration.
             */
            virtual void BeforeMigration() = 0;

            /**
             * @brief Preform all tasks need by the engine after migration.
             */
            virtual void AfterMigration() = 0;

            /**
             * @brief Preform all tasks need by the engine
             * before finalizing migration.
             */
            virtual void BeforeFinalize() = 0;

            /**
             * @brief Preform all tasks need by the engine
             * after finalizing migration.
             * @param[in] apMigrationData : MigrationData *
             */
            virtual operations::dataunits::MigrationData *AfterFinalize(
                    operations::dataunits::MigrationData *apMigrationData) = 0;

            virtual bool HasNextShot() = 0;

            virtual std::vector<uint> GetNextShot() = 0;

            /**
             * @brief Preform migration full cycle.
             * @return aMigrationData : MigrationData
             */
            operations::dataunits::MigrationData *Execute() {
                operations::dataunits::GridBox *gb = Initialize();
                BeforeMigration();
                while (HasNextShot()) {
                    mpEngine->MigrateShots(GetNextShot(), gb);
                    AfterMigration();
                }
                BeforeFinalize();
                operations::dataunits::MigrationData *finalized_data = AfterFinalize(mpEngine->Finalize(gb));
                return finalized_data;
            }

        protected:
            /// Engine instance needed by agent to preform task upon
            operations::engines::Engine *mpEngine{};

            /// CLI arguments
            int argc{};
            char **argv{};
        };
    }//namespace agents
}//namespace stbx

#endif //PIPELINE_AGENTS_AGENT_HPP
