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
// Created by zeyad-osama on 08/09/2020.
//

#ifndef OPERATIONS_LIB_ENGINES_ENGINE_HPP
#define OPERATIONS_LIB_ENGINES_ENGINE_HPP

#include "operations/helpers/callbacks/primitive/CallbackCollection.hpp"
#include "operations/data-units/concrete/migration/MigrationData.hpp"

#include <timer/Timer.h>

#include <vector>

namespace operations {
    namespace engines {
        /**
         * @note Each engine comes with it's own Timer and Logger. Logger
         * Channel should be initialized at each concrete implementation
         * and should be destructed at each destructor.
         */
        class Engine {
        public:
            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            virtual ~Engine() {};

            /**
             * @brief Initializes domain model.
             */
            virtual dataunits::GridBox *Initialize() = 0;

            /**
             * @brief The function that filters and returns all possible
             * shot ID in a vector to be fed to the migrate method.
             *
             * @return[out]
             * A vector containing all unique shot IDs.
             */
            virtual std::vector<uint> GetValidShots() = 0;
            /// @todo GetValidGathers

            /**
             * @brief The migration function that will apply the
             * reverse time migration process and produce the results needed.
             *
             * @param[in] shot_list
             * A vector containing the shot IDs to be migrated.
             */
            virtual void MigrateShots(std::vector<uint> shot_numbers, dataunits::GridBox *apGridBox) = 0;
            /// @todo ProcessGathers

            /**
             * @brief Finalizes and terminates all processes
             *
             * @return[out]
             * A float pointer to the array containing the final correlation result.
             */
            virtual dataunits::MigrationData *Finalize(dataunits::GridBox *apGridBox) = 0;

        protected:
            /// Callback collection to be called when not in release mode.
            helpers::callbacks::CallbackCollection *mpCallbacks;
            /// Computations parameters.
            common::ComputationParameters *mpParameters;
            /// Timer
            Timer *mpTimer;
        };
    } //namespace engines
} //namespace operations

#endif //OPERATIONS_LIB_ENGINES_ENGINE_HPP
