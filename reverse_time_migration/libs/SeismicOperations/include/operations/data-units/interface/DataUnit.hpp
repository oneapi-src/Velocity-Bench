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
// Created by zeyad-osama on 19/09/2020.
//

#ifndef OPERATIONS_LIB_DATA_UNIT_HPP
#define OPERATIONS_LIB_DATA_UNIT_HPP

namespace operations {
    namespace dataunits {

        enum REPORT_LEVEL {
            SIMPLE, VERBOSE
        };

        class DataUnit {
        public:
            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            virtual ~DataUnit() {};
        };
    } //namespace dataunits
} //namespace operations

#endif //OPERATIONS_LIB_DATA_UNIT_HPP