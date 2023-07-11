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
// Created by zeyad-osama on 25/09/2020.
//

#ifndef OPERATIONS_LIB_DATA_UNITS_MIGRATION_RESULT_HPP
#define OPERATIONS_LIB_DATA_UNITS_MIGRATION_RESULT_HPP

namespace operations {
    namespace dataunits {

        class Result {
        public:
            /**
             * @brief Constructor
             */
            explicit Result(float *data) {
                this->mpData = data;
            };

            /**
             * @brief Destructor.
             */
            ~Result() = default;

            float *GetData() {
                return this->mpData;
            }

            void SetData(float *data) {
                mpData = data;
            }

        private:
            float *mpData;
        };
    } //namespace dataunits
} //namespace operations

#endif //OPERATIONS_LIB_DATA_UNITS_MIGRATION_RESULT_HPP
