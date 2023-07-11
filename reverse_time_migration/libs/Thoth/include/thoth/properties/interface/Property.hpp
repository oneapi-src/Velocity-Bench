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
// Created by zeyad-osama on 02/11/2020.
//

#ifndef THOTH__CONFIGURATIONS_PROPERTY_HPP
#define THOTH__CONFIGURATIONS_PROPERTY_HPP

#include <string>
#include <map>

namespace thoth {
    namespace properties {
        /**
         * @brief
         */
        class Property {
        public:

            explicit Property(std::string &apPropertyKey, std::string &apPropertyValue) {
                this->mPropertyKey = apPropertyKey;
                this->mPropertyValue = apPropertyValue;
            }

            /**
             * @brief Destructor
             */
            virtual ~Property() = default;

        private:
            std::string mPropertyKey;
            std::string mPropertyValue;
        };

    } //properties
} //thoth

#endif //THOTH__CONFIGURATIONS_PROPERTY_HPP
