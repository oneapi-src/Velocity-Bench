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
// Created by amr-nasr on 07/06/2020.
//

#ifndef OPERATIONS_LIB_UTILS_WRITE_UTILS_H
#define OPERATIONS_LIB_UTILS_WRITE_UTILS_H

#include <operations/common/DataTypes.h>
#include <string>

namespace operations {
    namespace utils {
        namespace io {

            void write_adcig_segy(uint nx, uint ny, uint nz, uint nt,
                                  uint n_angles,
                                  float dx, float dy, float dz, float dt,
                                  const float *data,
                                  const std::string &file_name, bool is_traces);

            void write_segy(uint nx, uint ny, uint nz, uint nt,
                            float dx, float dy, float dz, float dt,
                            const float *data, const std::string &file_name, bool is_traces);

            void write_su(const float *temp, uint nx, uint nz,
                          const char *file_name, bool write_little_endian);

            void write_binary(float *temp, uint nx, uint nz, const char *file_name);
        } //namespace io
    } //namespace utils
} //namespace operations

#endif // OPERATIONS_LIB_UTILS_WRITE_UTILS_H
