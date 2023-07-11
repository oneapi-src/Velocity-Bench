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

#ifndef OPERATIONS_LIB_HELPERS_CALLBACKS_SEGY_WRITER_H
#define OPERATIONS_LIB_HELPERS_CALLBACKS_SEGY_WRITER_H

#include <operations/helpers/callbacks/interface/WriterCallback.h>

#include <string>

namespace operations {
    namespace helpers {
        namespace callbacks {

            class SegyWriter : public WriterCallback {
            public:
                SegyWriter(uint show_each, bool write_velocity, bool write_forward,
                           bool write_backward, bool write_reverse, bool write_migration,
                           bool write_re_extended_velocity,
                           bool write_single_shot_correlation, bool write_each_stacked_shot,
                           bool write_traces_raw, bool writer_traces_preprocessed,
                           const std::vector<std::string> &vec_params,
                           const std::vector<std::string> &vec_re_extended_params,
                           const std::string &write_path)
                        : WriterCallback(show_each, write_velocity, write_forward,
                                         write_backward, write_reverse, write_migration,
                                         write_re_extended_velocity,
                                         write_single_shot_correlation,
                                         write_each_stacked_shot,
                                         write_traces_raw, writer_traces_preprocessed,
                                         vec_params,
                                         vec_re_extended_params,
                                         write_path, "segy") {

                };

                std::string GetExtension() override;

                void WriteResult(uint nx, uint ny, uint nz, uint nt,
                                 float dx, float dy, float dz, float dt,
                                 float *data, std::string filename, bool is_traces) override;
            };
        } //namespace callbacks
    } //namespace operations
} //namespace operations

#endif // OPERATIONS_LIB_HELPERS_CALLBACKS_SEGY_WRITER_H
