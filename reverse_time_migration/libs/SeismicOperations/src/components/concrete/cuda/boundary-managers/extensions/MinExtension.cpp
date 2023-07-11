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

#include "operations/components/independents/concrete/boundary-managers/extensions/MinExtension.hpp"

#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <limits>

using namespace std;
using namespace operations::components;
using namespace operations::components::addons;
using namespace operations::dataunits;

void MinExtension::VelocityExtensionHelper(float *property_array,
                                           int start_x, int start_y, int start_z,
                                           int end_x, int end_y, int end_z,
                                           int nx, int ny, int nz,
                                           uint boundary_length) 
{
    std::cout << "MinExtension::VelocityExtensionHelper is not implemented" << std::endl;
    assert(0);

    /*!
     * change the values of velocities at boundaries (HALF_LENGTH excluded) to
     * zeros the start for x , y and z is at HALF_LENGTH and the end is at (nx -
     * HALF_LENGTH) or (ny - HALF_LENGTH) or (nz- HALF_LENGTH)
     */
    int nz_nx = nx * nz;
    //////float min_velocity = std::numeric_limits<float>::max; // TODO: Is this the same as MAXFLOAT?

    float *dev_min_velocity(nullptr);

    // In case of 2D
    if (ny == 1) {
        end_y = 1;
        start_y = 0;
    } else {
        std::cout << "3-D min extension not implemented" << std::endl;
        std::cout << "Aborted in " << __FILE__ << std::endl;
        assert(0);

        // Get maximum property_array value.
        //*!putting random values for velocities at the boundaries for y and with all
        // * x and z */
    }
    /*!putting random values for velocities at the boundaries for X and with all Y
     * and Z */

    /*!putting random values for velocities at the boundaries for z and with all x
     * and y */

    // Random-Corners in the boundaries nx-nz boundary intersection at bottom--
    // top boundaries not needed.

    // If 3-D, zero corners in the y-x and y-z plans.
    if (ny > 1) {
        std::cout << "3-D min extension not implemented" << std::endl;
        assert(0);
        // Random-Corners in the boundaries ny-nz boundary intersection at bottom--
        // top boundaries not needed.

        // Random-Corners in the boundaries nx-ny boundary intersection.
    }
}

void MinExtension::TopLayerExtensionHelper(float *property_array,
                                           int start_x, int start_z,
                                           int start_y, int end_x, int end_y,
                                           int end_z, int nx, int nz, int ny,
                                           uint boundary_length) {
    // Do nothing, no top layer to extend in random boundaries.
}

void MinExtension::TopLayerRemoverHelper(float *property_array, int start_x,
                                         int start_z, int start_y, int end_x,
                                         int end_y, int end_z, int nx,
                                         int nz, int ny,
                                         uint boundary_length) {
    // Do nothing, no top layer to remove in random boundaries.
}
