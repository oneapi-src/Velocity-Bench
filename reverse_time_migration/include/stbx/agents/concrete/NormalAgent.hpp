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

#ifndef PIPELINE_AGENTS_NORMAL_AGENT_HPP
#define PIPELINE_AGENTS_NORMAL_AGENT_HPP

#include <stbx/agents/interface/Agent.hpp>

namespace stbx {
    namespace agents {

/**
 * @brief
 */
        class NormalAgent : public Agent {

        public:
            /**
            * @brief Constructor.
            */
            NormalAgent() = default;

            /**
             * @brief Destructor.
             */
            ~NormalAgent() override;

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
            operations::dataunits::GridBox *Initialize() override;

            /**
             * @brief Preform all tasks need by the engine before migration.
             */
            void BeforeMigration() override;

            /**
             * @brief Preform all tasks need by the engine after migration.
             */
            void AfterMigration() override;

            /**
             * @brief Preform all tasks need by the engine
             * before finalizing migration.
             */
            void BeforeFinalize() override;

            /**
             * @brief Preform all tasks need by the engine
             * after finalizing migration.
             * @param[in] aMigrationData : MigrationData
             */
            operations::dataunits::MigrationData *AfterFinalize(
                    operations::dataunits::MigrationData *aMigrationData) override;

            bool HasNextShot() override;

            std::vector<uint> GetNextShot() override;

        private:
            uint mCount = 0;
        };
    }//namespace agents
}//namespace stbx

#endif //PIPELINE_AGENTS_NORMAL_AGENT_HPP
