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
// Created by zeyad-osama on 29/11/2020.
//

#ifndef THOTH_CONFIGURATIONS_CONFIGURATION_MAP_HPP
#define THOTH_CONFIGURATIONS_CONFIGURATION_MAP_HPP

#include <thoth//configurations/interface/MapKeys.h>

#include <string>
#include <sstream>

namespace thoth {
    namespace configuration {
        /**
         * @brief A configurations map containing a subset of properties.
         * It is organized as subsections, with each subsection containing a list of properties.
         */
        class ConfigurationMap {
        public:
            /**
             * @brief Default constructor.
             */
            ConfigurationMap() = default;

            /**
             * @brief Default destructor.
             */
            virtual ~ConfigurationMap() = default;

            /**
             * @brief Getter for the value of the property.
             * it is up to the user to give the correct type for the conversion.
             * If the key combination doesn't exist or has a null value,
             * the given default value is returned.
             *
             * @param[in] aSectionKey
             * The section key.
             *
             * @param[in] aPropertyKey
             * The property key.
             *
             * @param[in] aDefaultValue
             * The default value used if key doesn't not exist or is a null object.
             *
             * @return
             * T stored under this key, or the default value.
             */
            virtual float GetValue(const std::string &aSectionKey,
                                   const std::string &aPropertyKey,
                                   float aDefaultValue) = 0;

            virtual uint GetValue(const std::string &aSectionKey,
                                  const std::string &aPropertyKey,
                                  int aDefaultValue) = 0;

            virtual double GetValue(const std::string &aSectionKey,
                                    const std::string &aPropertyKey,
                                    double aDefaultValue) = 0;

            virtual std::string GetValue(const std::string &aSectionKey,
                                         const std::string &aPropertyKey,
                                         std::string const &aDefaultValue) = 0;

            virtual bool GetValue(const std::string &aSectionKey,
                                  const std::string &aPropertyKey,
                                  bool aDefaultValue) = 0;

            virtual void WriteValue(const std::string &aSectionKey,
                                    const std::string &aPropertyKey,
                                    std::string aValue) = 0;

            /**
             * @brief Check if a section entry exists in the object.
             *
             * @param[in] aSectionKey
             * The section key to check.
             *
             * @return True if the section exists.
             */
            virtual bool Contains(const std::string &aSectionKey) = 0;

            /**
             * @brief Check if a property entry exists in a specific section.
             *
             * @param[in] aSectionKey
             * The section key to check.
             *
             * @param[in] aPropertyKey
             * The property key to check
             *
             * @return True if the property in that section exists.
             */
            virtual bool Contains(const std::string &aSectionKey,
                                  const std::string &aPropertyKey) = 0;
        };
    } //namespace configurations
} //namespace operations

#endif //THOTH_CONFIGURATIONS_CONFIGURATION_MAP_HPP
