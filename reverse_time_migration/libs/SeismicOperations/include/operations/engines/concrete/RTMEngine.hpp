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
// Created by amr-nasr on 16/10/2019.
//

#ifndef OPERATIONS_LIB_ENGINES_RTM_ENGINE_HPP
#define OPERATIONS_LIB_ENGINES_RTM_ENGINE_HPP

#include <operations/engines/interface/Engine.hpp>
#include <operations/engine-configurations/concrete/RTMEngineConfigurations.hpp>

namespace operations {
    namespace engines {
        /**
         * @brief The RTM engine responsible of using the different
         * components to apply the reverse time migration (RTM).
         */
        class RTMEngine : public Engine {
        public:
            /**
             * @brief Constructor for the RTM engine giving it the configuration needed.
             *
             * @param[in] apConfiguration
             * The engine configuration that should be used for the RTM engine.
             *
             * @param[in] apParameters
             * The computation parameters that will control the simulations settings like
             * boundary length, order of numerical solution.
             */
            RTMEngine(configuration::RTMEngineConfigurations *apConfiguration,
                      common::ComputationParameters *apParameters);

            /**
             * @brief Constructor for the RTM engine giving it the configuration needed.
             *
             * @param[in] apConfiguration
             * The engine configuration that should be used for the RTM engine.
             *
             * @param[in] apParameters
             * The computation parameters that will control the simulations settings like
             * boundary length, order of numerical solution.
             *
             * @param[in] apCallbackCollection
             * The callback collection to be called throughout the execution if in debug
             * mode.
             */
            RTMEngine(configuration::RTMEngineConfigurations *apConfiguration,
                      common::ComputationParameters *apParameters,
                      helpers::callbacks::CallbackCollection *apCallbackCollection);

            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            ~RTMEngine() override;

            /**
             * @brief Initializes domain model.
             */
            dataunits::GridBox *
            Initialize() override;

            /**
             * @brief The function that filters and returns all possible
             * shot ID in a vector to be fed to the migrate method.
             *
             * @return[out]
             * A vector containing all unique shot IDs.
             */
            std::vector<uint>
            GetValidShots() override;

            /**
             * @brief The migration function that will apply the
             * reverse time migration process and produce the results needed.
             *
             * @param[in] shot_list
             * A vector containing the shot IDs to be migrated.
             */
            void
            MigrateShots(std::vector<uint> shot_numbers, dataunits::GridBox *apGridBox) override;

            /**
             * @brief The migration function that will apply the
             * reverse time migration process and produce the results needed.
             *
             * @param[in] shot_id
             * Shot IDs to be migrated.
             */
            void
            MigrateShots(uint shot_id, dataunits::GridBox *apGridBox);

            /**
             * @brief Finalizes and terminates all processes
             *
             * @return[out]
             * A float pointer to the array containing the final correlation result.
             */
            dataunits::MigrationData *
            Finalize(dataunits::GridBox *apGridBox) override;
            
            double GetIOReadTime() { return m_dReadIOTime; }
        private:
            /**
             * @brief Applies the forward propagation using the different
             * components provided in the configuration.
             */
            void
            Forward(dataunits::GridBox *apGridBox);

            /**
             * @brief Applies the backward propagation using the different
             * components provided in the configuration.
             */
            void
            Backward(dataunits::GridBox *apGridBox);

        private:
            /// The configuration containing the actual components to be used in the process.
            configuration::RTMEngineConfigurations *mpConfiguration;
            double m_dReadIOTime;
        };
    } //namespace engines
} //namespace operations

#endif // OPERATIONS_LIB_ENGINES_RTM_ENGINE_HPP
