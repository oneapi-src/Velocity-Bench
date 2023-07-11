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
// Created by amr-nasr on 11/11/2019.
//

#ifndef OPERATIONS_LIB_COMPONENTS_TRACE_WRITER_HPP
#define OPERATIONS_LIB_COMPONENTS_TRACE_WRITER_HPP

#include "operations/common/DataTypes.h"
#include "operations/components/independents/interface/Component.hpp"

namespace operations {
    namespace components {

        /**
         * @brief This class is used to Record at each time step the
         * pressure at the surface in the places we want to have receivers on.
         * Can be considered as a hydrophone.
         */
        class TraceWriter : public Component {
        public:
            /**
             * @brief Destructors should be overridden to ensure correct memory management.
             */
            virtual ~TraceWriter() {};

            /**
             * @brief Initializes the trace writer with all needed data for it
             * to be able to start recording traces according the the given configuration.
             *
             * @param[in] apModellingConfiguration
             * The modelling configuration to be followed in recording the traces.
             *
             * @param[in] output_file_name
             * The output file to write the traces into.
             * it is a trace file contains all the traces for only one who
             * defined by the source_point
             */
            virtual void InitializeWriter(ModellingConfiguration *apModellingConfiguration,
                                          std::string output_file_name) = 0;

            /**
             * @brief Records the traces from the domain according to the
             * configuration given in the initialize function.
             */
            virtual void RecordTrace() = 0;

        private:
            Component *Clone() override { return nullptr; } // Should never be called
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_TRACE_WRITER_HPP
