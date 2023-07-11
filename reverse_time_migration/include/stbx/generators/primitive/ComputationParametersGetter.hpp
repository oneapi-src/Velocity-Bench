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
// Created by marwan-elsafty on 21/01/2021.
//

#ifndef SEISMIC_TOOLBOX_COMPUTATIONPARAMETERSGETTER_HPP
#define SEISMIC_TOOLBOX_COMPUTATIONPARAMETERSGETTER_HPP

#include <stbx/generators/common/Keys.hpp>
#include <operations/common/DataTypes.h>
#include <libraries/nlohmann/json.hpp>


namespace stbx {
    namespace generators {
        struct Window {
            int use_window = 0;
            int left_win = DEF_VAL;
            int right_win = DEF_VAL;
            int front_win = DEF_VAL;
            int back_win = DEF_VAL;
            int depth_win = DEF_VAL;
        };

        struct StencilOrder {
            int order = DEF_VAL;
            HALF_LENGTH half_length = O_8;
        };


        class ComputationParametersGetter {

        public:
            explicit ComputationParametersGetter(const nlohmann::json &aMap);

            ~ComputationParametersGetter() = default;

            Window GetWindow();

            StencilOrder GetStencilOrder();

            int GetBoundaryLength();

            float GetSourceFrequency();

            float GetDTRelaxed();

            int GetBlock(const std::string &direction);

            int GetIsotropicCircle();

        private:
            nlohmann::json mMap;
        };

    };// namespace generators
}//namesapce stbx


#endif //SEISMIC_TOOLBOX_COMPUTATIONPARAMETERSGETTER_HPP