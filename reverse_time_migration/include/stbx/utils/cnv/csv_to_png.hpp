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
// Created by zeyad-osama on 05/01/2021.
//

#ifndef SEISMIC_TOOLBOX_UTILS_CNV_CSV_TO_PNG_HPP
#define SEISMIC_TOOLBOX_UTILS_CNV_CSV_TO_PNG_HPP

#if (USE_OpenCV)

#include <string>

namespace stbx {
    namespace utils {
        namespace cnv {

            int csv_to_png(const std::string &dir, float percentile);

        }//namespace cnv
    }//namespace utils
}//namespace stbx

#endif

#endif //SEISMIC_TOOLBOX_UTILS_CNV_CSV_TO_PNG_HPP
