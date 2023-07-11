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


#include "hip/hip_runtime.h"
//
// Created by amr-nasr on 18/11/2019.
//

#include "operations/components/independents/concrete/boundary-managers/extensions/ZeroExtension.hpp"
#include <fstream>


#include "Logging.h"

#include <cassert>

using namespace std;
using namespace operations::components;
using namespace operations::components::addons;
using namespace operations::dataunits;


__global__ void cuZeroVelocityExtension_X(float *property_array,
                                          int const start_x,
                                          int const start_y,
                                          int const start_z,
                                          int const nx,
                                          int const nz_nx,
                                          int const end_x
        )
{
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y * blockDim.y + threadIdx.y;
    int const z = blockIdx.z * blockDim.z + threadIdx.z;

    int const column = x; 
    int const depth  = y + start_y; 
    int const row    = z + start_z; 

    property_array[depth * nz_nx + row * nx + column + start_x]     = 0;
    property_array[depth * nz_nx + row * nx + (end_x - 1 - column)] = 0;
}


__global__ void ZeroVelocityExtension_Z(float      *property_array,
                                        int const   nx,
                                        int const   nz_nx,
                                        int const   start_x,
                                        int const   start_y,
                                        int const   start_z,
                                        int const   end_z
        )
{
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y * blockDim.y + threadIdx.y;
    int const z = blockIdx.z * blockDim.z + threadIdx.z;

    int column = x + start_x; 
    int depth  = y + start_y; 
    int row    = z; 

    /*!for values from y = HALF_LENGTH TO y = HALF_LENGTH +BOUND_LENGTH*/
    property_array[depth * nz_nx + (start_z + row) * nx + column] = 0;
    /*!for values from y = ny-HALF_LENGTH TO y =
     * ny-HALF_LENGTH-BOUND_LENGTH*/
    property_array[depth * nz_nx + (end_z - 1 - row) * nx + column] = 0;

}


void ZeroExtension::VelocityExtensionHelper(float *property_array,
                                            int start_x, int start_y, int start_z,
                                            int end_x, int end_y, int end_z,
                                            int nx, int ny, int nz,
                                            uint boundary_length) 
{
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
        std::cout << "3-D Zero extension not defined" << std::endl;
        assert(0);

        // general case for 3D
        /*!putting zero values for velocities at the boundaries for y and with all x
         * and z */
    }
    /*!putting zero values for velocities at the boundaries for X and with all Y
     * and Z */

    dim3 const cuBlockSize_ZeroExtendX(1, 1, 1), cuGridSize_ZeroExtendX(boundary_length, end_y - start_y, end_z - start_z);
    hipLaunchKernelGGL(cuZeroVelocityExtension_X, cuGridSize_ZeroExtendX, cuBlockSize_ZeroExtendX, 0, 0,
                              property_array,
                              start_x,
                              start_y,
                              start_z,
                              nx,
                              nz_nx,
                              end_x);
    
    checkLastHIPError();

    /*!putting zero values for velocities at the boundaries for z and with all x
     * and y */

    
    dim3 const cuBlockSize_ZeroExtendZ(1, 1, 1), cuGridSize_ZeroExtendZ(end_x - start_x, end_y - start_y, boundary_length);
    hipLaunchKernelGGL(ZeroVelocityExtension_Z, cuGridSize_ZeroExtendZ, cuBlockSize_ZeroExtendZ, 0, 0,
                            property_array,
                            nx,
                            nz_nx,
                            start_x,
                            start_y,
                            start_z,
                            end_z);

    checkLastHIPError();

}

void ZeroExtension::TopLayerExtensionHelper(float *property_array,
                                            int start_x, int start_y, int start_z,
                                            int end_x, int end_y, int end_z,
                                            int nx, int ny, int nz, uint boundary_length) {
    // Do nothing, no top layer to extend in random boundaries.
}

void ZeroExtension::TopLayerRemoverHelper(float *property_array, int start_x,
                                          int start_z, int start_y,
                                          int end_x, int end_y, int end_z,
                                          int nx, int nz, int ny,
                                          uint boundary_length) {
    // Do nothing, no top layer to remove in random boundaries.
}
