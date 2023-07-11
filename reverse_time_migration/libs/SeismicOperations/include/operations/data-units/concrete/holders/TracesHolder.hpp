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
// Created by zeyad-osama on 19/09/2020.
//

#ifndef OPERATIONS_LIB_DATA_UNITS_TRACES_HOLDER_HPP
#define OPERATIONS_LIB_DATA_UNITS_TRACES_HOLDER_HPP

#include "operations/data-units/interface/DataUnit.hpp"
#include "operations/common/DataTypes.h"

namespace operations {
    namespace dataunits {
        /**
         * @brief Class containing the available traces information.
         */
        class TracesHolder : public DataUnit {
        public:
            /**
             * @brief Constructor.
             */
            TracesHolder() 
                : Traces(nullptr)
                , PositionsX(nullptr)
                , PositionsY(nullptr)
                , ReceiversCountX(0)
                , ReceiversCountY(0)
                , TraceSizePerTimeStep(0)
                , SampleNT(0)
                , SampleDT(0.0f)

            {

            }

            /**
             * @brief Destructor.
             */
            ~TracesHolder() override = default;

        public:
            float *Traces;

            uint *PositionsX;
            uint *PositionsY;

            uint ReceiversCountX;
            uint ReceiversCountY;

            uint TraceSizePerTimeStep;

            uint SampleNT;
            float SampleDT;
        };
    } //namespace dataunits
} //namespace operations

#endif //OPERATIONS_LIB_DATA_UNITS_TRACES_HOLDER_HPP
