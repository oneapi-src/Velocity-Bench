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

#include "operations/components/independents/concrete/boundary-managers/extensions/RandomExtension.hpp"

#include <algorithm>
#include <cstdlib>
#include <cassert>

using namespace std;
using namespace operations::components;
using namespace operations::components::addons;
using namespace operations::dataunits;

/*
 * helper function because the SYCL::abs is not supported for now
 */
inline float _abs(float argA) {
    if (argA < 0) {
        argA = -argA;
    }
    return argA;
}

/*
 * helper function to randomize the contents of a buffer
 */
static void randomize(float *array, int size) {
    auto temp_arr = (float *) malloc(sizeof(float) * size);
    int i;

    for (i = 0; i < size; i++) {
        temp_arr[i] = (float) rand() / (float) (RAND_MAX);
    }
    free(temp_arr);
}

void RandomExtension::VelocityExtensionHelper(
        float *property_array,
        int start_x, int start_y, int start_z,
        int end_x, int end_y, int end_z,
        int nx, int ny, int nz,
        uint boundary_length) 
{
    std::cout << "RandomExtension::VelocityExtensionHelper not implemented" << std::endl; 
    assert(0);

    /*!
     * change the values of velocities at boundaries (HALF_LENGTH excluded) to
     * zeros the start for x , y and z is at HALF_LENGTH and the end is at (nx -
     * HALF_LENGTH) or (ny - HALF_LENGTH) or (nz- HALF_LENGTH)
     */
    int nz_nx = nx * nz;
    float max_velocity = 0;

    /*
     * create empty array of random values to be used to fill the property
     */
    unsigned long random_size;
    if (ny == 1) {
        random_size = (end_x + boundary_length) * (end_z + boundary_length) * 2;
    } else {
        random_size = (end_y + boundary_length) * (end_x + boundary_length) *
                      (end_z + boundary_length) * 2;
    }

    // In case of 2D
    if (ny == 1) {
        end_y = 1;
        start_y = 0;
        std::cout << "Aborted in " << __FILE__ << std::endl;
        assert(0);

    } else {
        std::cout << "3-D Random extension not implemented" << std::endl;
        assert(0);
        // Get maximum property_array value.
        // general case for 3D
/////        randomize(random_data, random_size);
/////        /*!putting random values for velocities at the boundaries for y and with all
/////         * x and z */
    }
    /////randomize(random_data, random_size);
    /*!putting random values for velocities at the boundaries for X and with all Y
     * and Z */
//////    randomize(random_data, random_size);
//////    /*!putting random values for velocities at the boundaries for z and with all x
//////     * and y */
//////    randomize(random_data, random_size);
//////    // Random-Corners in the boundaries nx-nz boundary intersection at bottom--
//////    // top boundaries not needed.

    /////randomize(random_data, random_size);
    // If 3-D, zero corners in the y-x and y-z plans.
    if (ny > 1) {
        std::cout << "3-D boundary extensions not supported" << std::endl;
        assert(0);
        // Random-Corners in the boundaries ny-nz boundary intersection at bottom--
        // top boundaries not needed.

        /////randomize(random_data, random_size);
        /////// Zero-Corners in the boundaries nx-ny boundary intersection on the top
        /////// layer--boundaries not needed.

        /////randomize(random_data, random_size);
        /////// Random-Corners in the boundaries nx-ny boundary intersection.
    }
}

void RandomExtension::TopLayerExtensionHelper(
        float *property_array,
        int start_x, int start_y, int start_z,
        int end_x, int end_y, int end_z,
        int nx, int ny, int nz,
        uint boundary_length) {
    // Do nothing, no top layer to extend in random boundaries.
}

void RandomExtension::TopLayerRemoverHelper(
        float *property_array,
        int start_x, int start_y, int start_z,
        int end_x, int end_y, int end_z,
        int nx, int ny, int nz,
        uint boundary_length) {
    // Do nothing, no top layer to remove in random boundaries.
}
