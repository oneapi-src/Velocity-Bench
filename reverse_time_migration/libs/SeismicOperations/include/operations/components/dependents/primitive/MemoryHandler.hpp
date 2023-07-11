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
// Created by zeyad-osama on 24/09/2020.
//

#ifndef OPERATIONS_LIB_COMPONENTS_MEMORY_HANDLER_HPP
#define OPERATIONS_LIB_COMPONENTS_MEMORY_HANDLER_HPP

#include "operations/components/independents/interface/Component.hpp"

#include "operations/common/DataTypes.h"

namespace operations {
    namespace components {

        class MemoryHandler : public DependentComponent {
        public:
            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            virtual ~MemoryHandler() {};

            /**
             * @brief Applies first touch initialization by accessing the given pointer in
             * the same way it will be accessed in the step function, and initializing the
             * values to 0.
             *
             * @param[in] ptr
             * The pointer that will be accessed for the first time.
             *
             * @param[in] apGridBox
             * As it contains all the meta data needed
             * i.e. nx, ny and nz or wnx, wny and wnz
             *
             * @param[in] enable_window
             * Lets first touch know which sizes to compute upon.
             * i.e. Grid size or window size.
             */
            virtual void FirstTouch(float *ptr, dataunits::GridBox *apGridBox, bool enable_window = false) = 0;

            MemoryHandler &operator=(MemoryHandler const &RHS) = delete;
        };
    }//namespace components
}//namespace operations

#endif //OPERATIONS_LIB_COMPONENTS_MEMORY_HANDLER_HPP
