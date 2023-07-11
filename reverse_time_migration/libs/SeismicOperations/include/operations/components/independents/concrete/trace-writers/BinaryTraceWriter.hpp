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
// Created by amr-nasr on 13/11/2019.
//

#ifndef OPERATIONS_LIB_COMPONENTS_TRACE_WRITERS_BINARY_TRACE_WRITER_HPP
#define OPERATIONS_LIB_COMPONENTS_TRACE_WRITERS_BINARY_TRACE_WRITER_HPP

#include <operations/components/independents/primitive/TraceWriter.hpp>
#include <operations/components/dependency/concrete/HasNoDependents.hpp>

namespace operations {
    namespace components {

        class BinaryTraceWriter : public TraceWriter, public dependency::HasNoDependents {
        public:
            explicit BinaryTraceWriter(operations::configuration::ConfigurationMap *apConfigurationMap);

            ~BinaryTraceWriter() override;

            void InitializeWriter(ModellingConfiguration *apModellingConfiguration,
                                  std::string output_filename) override;

            void RecordTrace() override;

            void SetComputationParameters(common::ComputationParameters *apParameters) override;

            void SetGridBox(dataunits::GridBox *apGridBox) override;

            void AcquireConfiguration() override;

        private:
            common::ComputationParameters *mpParameters = nullptr;

            dataunits::GridBox *mpGridBox = nullptr;

            std::ofstream *mpOutStream = nullptr;

            Point3D mReceiverStart;

            Point3D mReceiverEnd;

            Point3D mReceiverIncrement;

            BinaryTraceWriter           (BinaryTraceWriter const &RHS) = delete;
            BinaryTraceWriter &operator=(BinaryTraceWriter const &RHS) = delete;
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_TRACE_WRITERS_BINARY_TRACE_WRITER_HPP
