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
// Created by mirna-moawad on 1/16/20.
//

#ifndef OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_BOUNDARY_SAVER_H
#define OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_BOUNDARY_SAVER_H

#include <operations/common/ComputationParameters.hpp>
#include <operations/data-units/concrete/holders/GridBox.hpp>

namespace operations {
    namespace components {
        namespace helpers {

            void save_boundaries(dataunits::GridBox *apGridBox, common::ComputationParameters *apParameters,
                                 float *backup_boundaries, uint step, uint boundary_size);

            void restore_boundaries(dataunits::GridBox *apMainGrid, dataunits::GridBox *apInternalGrid,
                                    common::ComputationParameters *apParameters,
                                    const float *backup_boundaries, uint step, uint boundary_size);

        }//namespace helpers
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_FORWARD_COLLECTORS_BOUNDARY_SAVER_H
