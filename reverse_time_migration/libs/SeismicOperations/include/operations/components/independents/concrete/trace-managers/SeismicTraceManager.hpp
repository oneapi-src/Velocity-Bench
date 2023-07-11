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
// Created by ingy-mounir on 1/28/20.
//

#ifndef OPERATIONS_LIB_COMPONENTS_TRACE_MANAGERS_SEISMIC_TRACE_MANAGER_HPP
#define OPERATIONS_LIB_COMPONENTS_TRACE_MANAGERS_SEISMIC_TRACE_MANAGER_HPP

#include <operations/components/independents/primitive/TraceManager.hpp>
#include <operations/components/dependency/concrete/HasNoDependents.hpp>
#include <operations/data-units/concrete/holders/FrameBuffer.hpp>

#include <thoth/api/thoth.hpp>

#include <fstream>
#include <unordered_map>

namespace operations {
    namespace components {

        class SeismicTraceManager : public TraceManager,
                                    public dependency::HasNoDependents {
        public:
            SeismicTraceManager(operations::configuration::ConfigurationMap *apConfigurationMap);

            ~SeismicTraceManager() override;

            void SetComputationParameters(common::ComputationParameters *apParameters) override;

            void SetGridBox(dataunits::GridBox *apGridBox) override;

            void ReadShot(std::vector<std::string> file_names, uint shot_number,
                          std::string sort_key) override;

            void PreprocessShot(uint cut_off_time_step) override;

            void ApplyTraces(uint time_step) override;

            void ApplyIsotropicField() override;

            void RevertIsotropicField() override;

            dataunits::TracesHolder *GetTracesHolder() override;

            Point3D *GetSourcePoint() override;

            std::vector<uint> GetWorkingShots(std::vector<std::string> file_names,
                                              uint min_shot, uint max_shot, std::string type) override;

            void AcquireConfiguration() override;

        private:
            Point3D SDeLocalizePoint(Point3D point, bool is_2D, uint half_length,
                                     uint bound_length);

            IPoint3D DeLocalizePointS(IPoint3D aIPoint3D, bool is_2D,
                                      uint half_length, uint bound_length);


        private:
            common::ComputationParameters *mpParameters = nullptr;

            dataunits::GridBox *mpGridBox = nullptr;

            Point3D mpSourcePoint;

            thoth::streams::Reader *mpSeismicReader = nullptr;

            INTERPOLATION mInterpolation;

            dataunits::TracesHolder *mpTracesHolder = nullptr;

            dataunits::FrameBuffer<float> mpDTraces;

            dataunits::FrameBuffer <uint> mpDPositionsY;

            dataunits::FrameBuffer <uint> mpDPositionsX;

            float mTotalTime;

            int mShotStride;

            SeismicTraceManager           (SeismicTraceManager const &RHS) = delete;
            SeismicTraceManager &operator=(SeismicTraceManager const &RHS) = delete;
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_TRACE_MANAGERS_SEISMIC_TRACE_MANAGER_HPP
