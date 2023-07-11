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
// Created by amr-nasr on 18/11/2019.
//

#include "operations/components/independents/concrete/boundary-managers/extensions/HomogenousExtension.hpp"
#include <cassert>

using namespace std;
using namespace operations::components;
using namespace operations::components::addons;
using namespace operations::dataunits;

HomogenousExtension::HomogenousExtension(bool use_top_layer) {
    this->mUseTop = use_top_layer;
}

void HomogenousExtension::VelocityExtensionHelper(float *property_array,
                                                  int start_x, int start_y, int start_z,
                                                  int end_x, int end_y, int end_z,
                                                  int nx, int ny, int nz,
                                                  uint boundary_length) 
{
    std::cout << "Aborted in " << __FILE__ << std::endl;
    assert(0);

    /*!
     * change the values of velocities at boundaries (HALF_LENGTH excluded) to
     * zeros the start for x , y and z is at HALF_LENGTH and the end is at (nx -
     * HALF_LENGTH) or (ny - HALF_LENGTH) or (nz- HALF_LENGTH)
     */
    int nz_nx = nx * nz;

    // In case of 2D
    if (ny == 1) {
        end_y = 1;
        start_y = 0;
    } else {
        std::cout << "3-D Homogeneous Extension not implemented" << std::endl;
        assert(0);

        // general case for 3D
        /*!putting the nearest property_array adjacent to the boundary as the value
         * for all velocities at the boundaries for y and with all x and z */
    }

    /*!putting the nearest property_array adjacent to the boundary as the value
     * for all velocities at the boundaries for x and with all z and y */

    std::cout << "Homogeneous extension not implemented" << std::endl;
    assert(0);
}

void HomogenousExtension::TopLayerExtensionHelper(float *property_array,
                                                  int start_x, int start_y, int start_z,
                                                  int end_x, int end_y, int end_z,
                                                  int nx, int ny, int nz, uint boundary_length) {
    if (this->mUseTop) {
        int nz_nx = nx * nz;
        if (ny == 1) {
            start_y = 0;
            end_y = 1;
        }
        /*!putting the nearest property_array adjacent to the boundary as the value
         * for all velocities at the boundaries for z and with all x and y */
        std::cout << "TopLayerExtensionHelper for Homogeneous Extension is not implemented " << std::endl;
        assert(0);
    }
}

void HomogenousExtension::TopLayerRemoverHelper(float *property_array,
                                                int start_x, int start_y, int start_z,
                                                int end_x, int end_y, int end_z,
                                                int nx, int ny, int nz, uint boundary_length) {
    if (this->mUseTop) {
        if (ny == 1) {
            start_y = 0;
            end_y = 1;
        }
        int nz_nx = nx * nz;
        /*!putting the nearest property_array adjacent to the boundary as the value
         * for all velocities at the boundaries for z and with all x and y */
        std::cout << "HomogeneousExtension::TopLayerRemoverHelper is not implemented" << std::endl;
        assert(0);
    }
}
