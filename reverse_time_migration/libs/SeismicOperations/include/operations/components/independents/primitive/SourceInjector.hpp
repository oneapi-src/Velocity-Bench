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

#ifndef OPERATIONS_LIB_COMPONENTS_SOURCE_INJECTOR_HPP
#define OPERATIONS_LIB_COMPONENTS_SOURCE_INJECTOR_HPP

#include "operations/components/independents/interface/Component.hpp"

#include "operations/common/DataTypes.h"

namespace operations {
    namespace components {

        /**
         * @brief Source Injector Interface. All concrete techniques for source
         * injection should be implemented using this interface.
         */
        class SourceInjector : public Component {
        public:
            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            virtual ~SourceInjector() {};

            /**
             * @brief Sets the source point to apply the injection to it.
             *
             * @param[in] source_point
             * The designated source point.
             */
            virtual void SetSourcePoint(Point3D *source_point) = 0;

            /**
             * @brief Apply source injection to the wave field(s). It should inject
             * the current frame in our grid with the appropriate value. It should be
             * responsible of controlling when to stop the injection.
             *
             * @param[in] time_step
             * The time step corresponding to the current frame. This should be 1-based,
             * meaning the first frame that should be injected with our source should have
             * a time_step equal to 1.
             */
            virtual void ApplySource(uint time_step) = 0;

            /**
             * @brief Applies Isotropic Field upon all parameters' models
             *
             * @see ComputationParameters->GetIsotropicRadius()
             */
            virtual void ApplyIsotropicField() = 0;

            /**
             * @brief Reverts Isotropic Field upon all parameters' models
             *
             * @see ComputationParameters->GetIsotropicRadius()
             */
            virtual void RevertIsotropicField() = 0;

            /**
             * @brief Gets the time step that the source injection would stop after.
             *
             * @return[out]
             * An unsigned integer indicating the time step at which the source injection
             * would stop.
             */
            virtual uint GetCutOffTimeStep() = 0;

        private:
            Component *Clone() override { return nullptr; } // Should never be called
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_SOURCE_INJECTOR_HPP
