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

#ifndef OPERATIONS_LIB_COMPONENTS_COMPUTATION_KERNEL_HPP
#define OPERATIONS_LIB_COMPONENTS_COMPUTATION_KERNEL_HPP

#include "operations/components/independents/interface/Component.hpp"

#include "operations/components/dependents/primitive/MemoryHandler.hpp"
#include "BoundaryManager.hpp"

#include "operations/common/DataTypes.h"

namespace operations {
    namespace components {

        /**
         * @brief Computation Kernel Interface. All concrete techniques for
         * solving different wave equations should be implemented using this interface.
         */
        class ComputationKernel : public Component {
        public:
            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            virtual ~ComputationKernel() {};

            /**
             * @brief This function should solve the wave equation.  It calculates the next
             * time step from previous and current frames.
             * <br>
             * It should also update the GridBox structure so that after the function call,
             * the GridBox structure current frame should point to the newly calculated result,
             * the previous frame should point to the current frame at the time of the
             * function call. The next frame should point to the previous frame at the time of
             * the function call.
             */
            virtual void Step() = 0;

            /**
             * @brief Set kernel boundary manager to be used and called internally.
             *
             * @param[in] apBoundaryManager
             * The boundary manager to be used.
             */
            void SetBoundaryManager(BoundaryManager *apBoundaryManager) {
                this->mpBoundaryManager = apBoundaryManager;
            }

            /**
             * @return[out] The memory handler used.
             */
            virtual MemoryHandler *GetMemoryHandler() {
                return this->mpMemoryHandler;
            }

            virtual void SetAdjoint(bool aAdjoint) {
                this->mAdjoint = aAdjoint;
            }

        protected:
            /// Boundary Manager instance to be used by the step function.
            BoundaryManager *mpBoundaryManager;

            /// Memory Handler instance to be used for all memory
            /// handling (i.e. First touch)
            MemoryHandler *mpMemoryHandler;

            bool mAdjoint;
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_COMPUTATION_KERNEL_HPP
