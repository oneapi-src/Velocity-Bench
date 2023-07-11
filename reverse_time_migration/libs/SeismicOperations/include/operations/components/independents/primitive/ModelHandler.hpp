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

#ifndef OPERATIONS_LIB_COMPONENTS_MODEL_HANDLER_HPP
#define OPERATIONS_LIB_COMPONENTS_MODEL_HANDLER_HPP

#include "operations/components/independents/interface/Component.hpp"

#include "operations/common/DataTypes.h"
#include <operations/components/independents/primitive/ComputationKernel.hpp>

#include <string>
#include <vector>
#include <map>

namespace operations {
    namespace components {

        /**
         * @brief Model Handler Interface. All concrete techniques for loading
         * or setting models should be implemented using this interface.
         */
        class ModelHandler : public Component {
        public:
            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            virtual ~ModelHandler() {};

            /**
             * @brief This function should do the following:
             * <ul>
             *      <li>
             *      Create a new GridBox with all its frames
             *      (previous, current, next) allocated in memory.
             *      </li>
             *      <li>
             *      The models are loaded into the frame.
             *      </li>
             *      <li>
             *      This should account for the allocation of the boundaries around
             *      our domain(top, bottom, left, right).
             *      </li>
             *      <li>
             *      It should set all possible GridBox properties to be ready for
             *      actual computations.
             *      </li>
             * </ul>
             *
             * @param[in] files_names
             * The filenames vector of the files containing the model, the first filename
             * in the vector should be the velocity file, the second should be the density
             * file name.
             *
             * @param[in] apComputationKernel
             * The computation kernel to be used for first touch.
             *
             * @return[out]
             * GridBox object that was allocated, and setup appropriately.
             */
            virtual dataunits::GridBox *ReadModel(std::map<std::string, std::string> const &files_names) = 0;

            /**
             * @brief All pre-processing needed to be done on the model before the
             * beginning of the reverse time migration, should be applied in this function.
             */
            virtual void PreprocessModel() = 0;

            /**
             * @brief Setup the window properties if needed by copying the
             * needed window from the full model.
             */
            virtual void SetupWindow() = 0;

        private:
            Component *Clone() override { return nullptr; } // Should never be called
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_MODEL_HANDLER_HPP
