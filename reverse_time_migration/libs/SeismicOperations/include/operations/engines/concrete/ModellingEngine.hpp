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
// Created by amr-nasr on 12/11/2019.
//

#ifndef OPERATIONS_LIB_ENGINES_MODELLING_ENGINE_HPP
#define OPERATIONS_LIB_ENGINES_MODELLING_ENGINE_HPP

#include "operations/engines/interface/Engine.hpp"

#include "operations/engine-configurations/concrete/ModellingEngineConfigurations.hpp"
#include "operations/helpers/callbacks/primitive/CallbackCollection.hpp"

#include <timer/Timer.h>

namespace operations {
    namespace engines {
        /**
         * @brief The Modelling engine responsible of using the different
         * components to model the needed algorithm.
         */
        class ModellingEngine : public Engine {
        public:
            /**
             * @brief Constructor to start the modelling engine given the appropriate engine
             * configuration.
             *
             * @param[in] apConfiguration
             * The configuration which will control the actual work of the engine.
             *
             * @param[in] apParameters
             * The computation parameters that will control the simulations settings like
             * boundary length, order of numerical solution.
             */
            ModellingEngine(configuration::ModellingEngineConfigurations *apConfiguration,
                            common::ComputationParameters *apParameters);

            /**
             * @brief Constructor to start the modelling engine given the appropriate engine
             * configuration.
             *
             * @param[in] apConfiguration
             * The configuration which will control the actual work of the engine.
             *
             * @param[in] apParameters
             * The computation parameters that will control the simulations settings like
             * boundary length, order of numerical solution.
             *
             * @param[in] apCallbackCollection
             * The callbacks registered to be called in the right time.
             */
            ModellingEngine(configuration::ModellingEngineConfigurations *apConfiguration,
                            common::ComputationParameters *apParameters,
                            helpers::callbacks::CallbackCollection *apCallbackCollection);

            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            ~ModellingEngine() override;

            ModellingEngine &operator=(ModellingEngine const &RHS) = delete;
            ModellingEngine           (ModellingEngine const &RHS) = delete;

            /**
             * @brief Run the initialization steps for the modelling engine.
             *
             * @return[out] GridBox
             */
            dataunits::GridBox *Initialize() override;

            /**
             * @brief The function that filters and returns all possible
             * shot ID in a vector to be fed to the migrate method.
             *
             * @return[out]
             * A vector containing all unique shot IDs.
             */
            std::vector<uint> GetValidShots() override;

            /**
             * @brief The migration function that will apply the
             * reverse time migration process and produce the results needed.
             *
             * @param[in] shot_list
             * A vector containing the shot IDs to be migrated.
             */
            void MigrateShots(std::vector<uint> shot_numbers, dataunits::GridBox *apGridBox) override;

            /**
             * @brief Finalizes and terminates all processes
             *
             * @return[out]
             * A float pointer to the array containing the final correlation result.
             */
            dataunits::MigrationData *Finalize(dataunits::GridBox *apGridBox) override;


        private:
            /**
             * @brief Applies the forward propagation using the different
             * components provided in the configuration.
             */
            void Forward(dataunits::GridBox *apGridBox, std::vector<uint> const &aShotNumbers);

        private:
            ///The configuration containing the actual components to be used in the process.
            configuration::ModellingEngineConfigurations *mpConfiguration;
            /// Callback collection to be called when not in release mode.
            operations::helpers::callbacks::CallbackCollection *mpCallbacks;
            /// The modelling configuration.
            ModellingConfiguration mpModellingConfiguration;
            /// Computations parameters.
            common::ComputationParameters *mpParameters;
        };
    } //namespace engines
} //namespace operations

#endif // OPERATIONS_LIB_ENGINES_MODELLING_ENGINE_HPP
