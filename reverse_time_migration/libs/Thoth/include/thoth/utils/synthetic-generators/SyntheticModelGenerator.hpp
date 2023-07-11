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
// Created by pancee on 12/28/20.
//

#ifndef THOTH_UTILS_SYNTHETIC_GENERATORS_SYNTHETIC_MODEL_GENERATOR_HPP
#define THOTH_UTILS_SYNTHETIC_GENERATORS_SYNTHETIC_MODEL_GENERATOR_HPP

#include <thoth/data-units/concrete/Gather.hpp>
#include <thoth/configurations/interface/Configurable.hpp>

#include <libraries/nlohmann/json.hpp>

namespace thoth {
    namespace generators {
        class SyntheticModelGenerator : public thoth::configuration::Configurable {

        public:
            /**
             * @brief Constructor
             *
             * @param apConfigurationMap
             */
            explicit SyntheticModelGenerator(
                    thoth::configuration::ConfigurationMap *apConfigurationMap);

            /**
            * @brief Destructor
            */
            ~SyntheticModelGenerator() override;

            void
            AcquireConfiguration() override;

            /**
             * @brief Generate synthetic model
             */
            void Generate();

            /**
             * @brief Get pointer to generated model
             */
            inline float
            *GetGeneratedModel() {
                return this->GeneratedTracesData;
            }

            /**
             * @brief Get meta data map
             */
            inline nlohmann::json GetMetaData() {
                return this->mMetaData;
            }

            /**
             * @brief Build gather from generated synthetic model data, and set trace headers
             */
            void
            BuildGather();

            /**
             * @brief Get gather
             */
            inline thoth::dataunits::Gather
            *GetGather() {
                return &this->mModel;
            }

        private:
            void
            GenerateModel();

            void
            GenerateLayers();

            void
            InjectSaltBodies();

            void
            InjectCracks();

        private:
            /// Meta data file path
            std::string mMetaDataFilepath;

            thoth::configuration::ConfigurationMap *mpConfigurationMap;

            /// Generated traces data
            float *GeneratedTracesData;

            //// Gather built from GeneratedTracesData
            thoth::dataunits::Gather mModel;

            /// Metadata map
            nlohmann::json mMetaData;

            SyntheticModelGenerator           (SyntheticModelGenerator const &RHS) = delete;
            SyntheticModelGenerator &operator=(SyntheticModelGenerator const &RHS) = delete;
        };
    } //generators
} //thoth


#endif //THOTH_UTILS_SYNTHETIC_GENERATORS_SYNTHETIC_MODEL_GENERATOR_HPP
