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

#ifndef OPERATIONS_LIB_COMPONENTS_TRACE_MANAGERS_BINARY_TRACE_MANAGER_HPP
#define OPERATIONS_LIB_COMPONENTS_TRACE_MANAGERS_BINARY_TRACE_MANAGER_HPP

#include <operations/components/independents/primitive/TraceManager.hpp>
#include <operations/components/dependency/concrete/HasNoDependents.hpp>

#include <fstream>

namespace operations {
    namespace components {

        class BinaryTraceManager : public TraceManager,
                                   public dependency::HasNoDependents {
        public:
            explicit BinaryTraceManager(operations::configuration::ConfigurationMap *apConfigurationMap);

            ~BinaryTraceManager() override;

            void ReadShot(std::vector<std::string> filenames,
                          uint shot_number, std::string sort_key) override;

            void PreprocessShot(uint cut_off_time_step) override;

            void ApplyTraces(uint time_step) override;

            void ApplyIsotropicField() override;

            void RevertIsotropicField() override;

            dataunits::TracesHolder *GetTracesHolder() override;

            void SetComputationParameters(common::ComputationParameters *apParameters) override;

            void SetGridBox(dataunits::GridBox *apGridBox) override;

            Point3D *GetSourcePoint() override;

            std::vector<uint> GetWorkingShots(std::vector<std::string> filenames,
                                              uint min_shot, uint max_shot,
                                              std::string type) override;

            void AcquireConfiguration() override;

        private:
            Point3D DeLocalizePoint(Point3D point, bool is_2D,
                                    uint half_length, uint bound_length);


        private:
            BinaryTraceManager(BinaryTraceManager const &RHS) = delete;
            BinaryTraceManager &operator=(BinaryTraceManager const &RHS) = delete;

            common::ComputationParameters *mpParameters = nullptr;

            dataunits::GridBox *mpGridBox = nullptr;

            Point3D mSourcePoint;

            INTERPOLATION mInterpolation;

            dataunits::TracesHolder *mpTracesHolder = nullptr;

            Point3D mReceiverStart;

            Point3D mReceiverEnd;

            Point3D mReceiverIncrement;

            uint mAbsoluteShotNumber;

            float mTotalTime;

            dataunits::FrameBuffer<float> mpDTraces;
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_TRACE_MANAGERS_BINARY_TRACE_MANAGER_HPP
