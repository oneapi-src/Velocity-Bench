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
// Created by zeyad-osama on 10/01/2021.
//

#ifndef OPERATIONS_LIB_UTILS_COMPRESSORS_COMPRESSOR_HPP
#define OPERATIONS_LIB_UTILS_COMPRESSORS_COMPRESSOR_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace operations {
    namespace utils {
        namespace compressors {
            class Compressor {
            public:
                Compressor() = default;

                ~Compressor() = default;

                /**
                 * @brief This is the main-entry point for all compression algorithms.
                 * Currently using a naive switch based differentiation of algorithm used
                 */
                static void Compress(float *array, int nx, int ny, int nz, int nt, double tolerance,
                                     unsigned int codecType, const char *filename,
                                     bool zfp_is_relative);

                /**
                 * @brief This is the main-entry point for all decompression algorithms.
                 * Currently using a naive switch based differentiation of algorithm used
                 */
                static void Decompress(float *array, int nx, int ny, int nz, int nt, double tolerance,
                                       unsigned int codecType, const char *filename,
                                       bool zfp_is_relative);
            };
        } //namespace compressors
    } //namespace utils
} //namespace operations

#endif //OPERATIONS_LIB_UTILS_COMPRESSORS_COMPRESSOR_HPP
