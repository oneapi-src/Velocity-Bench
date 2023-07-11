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
// Created by amr-nasr on 12/11/2019.
//

#ifndef OPERATIONS_LIB_HELPERS_CALLBACKS_WRITER_CALLBACK_H
#define OPERATIONS_LIB_HELPERS_CALLBACKS_WRITER_CALLBACK_H

#include <operations/helpers/callbacks/interface/Callback.hpp>

#include <string>
#include <vector>

namespace operations {
    namespace helpers {
        namespace callbacks {

            class WriterCallback : public Callback {
            public:
                WriterCallback(uint show_each, bool write_params, bool write_forward,
                               bool write_backward, bool write_reverse, bool write_migration,
                               bool write_re_extended_params,
                               bool write_single_shot_correlation, bool write_each_stacked_shot,
                               bool write_traces_raw, bool writer_traces_preprocessed,
                               const std::vector<std::string> &vec_params,
                               const std::vector<std::string> &vec_re_extended_params,
                               const std::string &write_path, const std::string &folder_name);

                virtual std::string GetExtension() = 0;

                virtual void WriteResult(uint nx, uint ny, uint nz, uint nt,
                                         float dx, float dy, float dz, float dt,
                                         float *data, std::string filename, bool is_traces) = 0;

                void BeforeInitialization(common::ComputationParameters *apParameters) override;

                void AfterInitialization(dataunits::GridBox *apGridBox) override;

                void BeforeShotPreprocessing(dataunits::TracesHolder *traces) override;

                void AfterShotPreprocessing(dataunits::TracesHolder *traces) override;

                void BeforeForwardPropagation(dataunits::GridBox *apGridBox) override;

                void AfterForwardStep(dataunits::GridBox *box, uint time_step) override;

                void BeforeBackwardPropagation(dataunits::GridBox *apGridBox) override;

                void AfterBackwardStep(dataunits::GridBox *apGridBox, uint time_step) override;

                void AfterFetchStep(dataunits::GridBox *apGridBox, uint time_step) override;

                void BeforeShotStacking(dataunits::GridBox *apGridBox,
                                        dataunits::FrameBuffer<float> *shot_correlation) override;

                void AfterShotStacking(dataunits::GridBox *apGridBox,
                                       dataunits::FrameBuffer<float> *stacked_shot_correlation) override;

                void AfterMigration(dataunits::GridBox *apGridBox,
                                    dataunits::FrameBuffer<float> *stacked_shot_correlation) override;

            private:
                uint mShowEach;
                uint mShotCount;
                bool mIsWriteParams;
                bool mIsWriteForward;
                bool mIsWriteBackward;
                bool mIsWriteReverse;
                bool mIsWriteMigration;
                bool mIsWriteReExtendedParams;
                bool mIsWriteSingleShotCorrelation;
                bool mIsWriteEachStackedShot;
                bool mIsWriteTracesRaw;
                bool mIsWriteTracesPreprocessed;
                std::string mWritePath;
                std::vector<std::string> mParamsVec;
                std::vector<std::string> mReExtendedParamsVec;
            };
        } //namespace callbacks
    } //namespace operations
} //namespace operations

#endif // OPERATIONS_LIB_HELPERS_CALLBACKS_WRITER_CALLBACK_H
