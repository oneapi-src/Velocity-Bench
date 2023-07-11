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

#ifndef OPERATIONS_LIB_COMPONENTS_TRACE_MANAGER_HPP
#define OPERATIONS_LIB_COMPONENTS_TRACE_MANAGER_HPP

#include "operations/components/independents/interface/Component.hpp"

#include "operations/common/DataTypes.h"
#include "operations/data-units/concrete/holders/TracesHolder.hpp"

#include <string>
#include <vector>

namespace operations {
    namespace components {

        /**
         * @brief Trace Manager Interface.
         * All concrete techniques for reading, pre-processing and injecting
         * the traces corresponding to a model should be implemented using
         * this interface.
         */
        class TraceManager : public Component {
        public:
            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            virtual ~TraceManager() {};

            /**
             * @brief Function that should read the traces for a specified shot,
             * and form the trace grid appropriately.
             *
             * @param[in] files_names
             * A vector of files' names containing all the shots files.
             *
             * @param[in] shot_number
             * The shot number or id of the shot that its traces are targeted to be read.
             *
             * @param[in] sort_key
             * The type of sorting to access the data in.
             */
            virtual void ReadShot(std::vector<std::string> files_names,
                                  uint shot_number, std::string sort_key) = 0;

            /**
             * @brief Function that should be possible for the pre-processing of the shot traces
             * already read. Pre-processing includes interpolation of the traces, any type
             * of muting or special pre-processing applied on the traces and setting the
             * number of time steps of the simulation appropriately. Any changes needed to
             * be applied on the meta data of the grid should be done.
             *
             * @param[in] cut_off_time
             * The time step at which the source injection ends.
             */
            virtual void PreprocessShot(uint cut_off_time) = 0;

            /**
             * @brief Function that injects traces into the current frame of our GridBox
             * according to the time step to be used. this function is used in the
             * backward propagation
             *
             * @param[in] time_step
             * The times_step of the current frame in our GridBox object.
             * Starts from nt - 1 with this being the time of the first trace
             * i.e. (backward propagation starts from nt - 1)
             */
            virtual void ApplyTraces(uint time_step) = 0;

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
             * @brief Getter to the property containing the shot location source
             * point of the current read traces.
             * This value should be set in the ReadShot function.
             *
             * @return[out]
             * A pointer to a Point3D that indicates the source point.
             * Point3D is a Point coordinates in 3D.
            */
            virtual Point3D *GetSourcePoint() = 0;

            /**
             * @return[out]
             * Pointer containing the information and values of the traces to be injected.
             * Dimensions should be in this order as follows:
             * <ul>
             *      <li>
             *      majority[time][y-dimension][x-dimension].
             *      </li>
             *      <li>
             *      Traces: Structure containing the available traces information.
             *      </li>
             * <ul>
            */
            virtual dataunits::TracesHolder *GetTracesHolder() = 0;

            /**
             * @brief Get the shots that you can be migrated from the data according
             * to a min/max and type.
             *
             * @param[in] files_names
             * Files containing the traces.
             *
             * @param[in] min_shot
             * Minimum shot ID to allow.
             *
             * @param[in] max_shot
             * Maximum shot ID to allow.
             *
             * @param[in] type
             * Can be "CDP" or "CSR"
             *
             * @return[out]
             * Vector containing the shot IDs.
             */
            virtual std::vector<uint> GetWorkingShots(std::vector<std::string> files_names,
                                                      uint min_shot, uint max_shot,
                                                      std::string type) = 0;

        private:
            Component *Clone() override { return nullptr; } // Should never be called

            Component &operator=(Component const &RHS) = delete;
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_TRACE_MANAGER_HPP
