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
#include <operations/backend/OneAPIBackend.hpp>

using namespace sycl;
using namespace std;
using namespace operations::components;
using namespace operations::components::addons;
using namespace operations::dataunits;
using namespace operations::backend;

HomogenousExtension::HomogenousExtension(bool use_top_layer) {
    this->mUseTop = use_top_layer;
}

void HomogenousExtension::VelocityExtensionHelper(float *property_array,
                                                  int start_x, int start_y, int start_z,
                                                  int end_x, int end_y, int end_z,
                                                  int nx, int ny, int nz,
                                                  uint boundary_length) {
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
        // general case for 3D
        /*!putting the nearest property_array adjacent to the boundary as the value
         * for all velocities at the boundaries for y and with all x and z */
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            auto global_range =
                    range<3>(end_x - start_x, boundary_length, end_z - start_z);
            auto local_range = range<3>(1, 1, 1);
            auto global_nd_range = nd_range<3>(global_range, local_range);

            cgh.parallel_for<class Homogenous_velocity_extension_Y>(
                    global_nd_range, [=](nd_item<3> it) {
                        int column = it.get_global_id(0) + start_x;
                        int depth = it.get_global_id(1);
                        int row = it.get_global_id(2) + start_z;

                        /*!for values from y = HALF_LENGTH TO y= HALF_LENGTH +BOUND_LENGTH*/
                        int p_idx = (depth + start_y) * nz_nx + row * nx + column;
                        int p2_idx =
                                (boundary_length + start_y) * nz_nx + row * nx + column;
                        property_array[p_idx] = property_array[p2_idx];

                        /*!for values from y = ny-HALF_LENGTH TO y =
                         * ny-HALF_LENGTH-BOUND_LENGTH*/
                        p_idx = (end_y - 1 - depth) * nz_nx + row * nx + column;
                        p2_idx = (end_y - 1 - boundary_length) * nz_nx + row * nx + column;
                        property_array[p_idx] = property_array[p2_idx];
                    });
        });
    }

    /*!putting the nearest property_array adjacent to the boundary as the value
     * for all velocities at the boundaries for x and with all z and y */
    OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
        auto global_range =
                range<3>(boundary_length, end_y - start_y, end_z - start_z);
        auto local_range = range<3>(1, 1, 1);
        auto global_nd_range = nd_range<3>(global_range, local_range);

        cgh.parallel_for<class Homogenous_velocity_extension_X>(
                global_nd_range, [=](nd_item<3> it) {
                    int column = it.get_global_id(0);
                    int depth = it.get_global_id(1) + start_y;
                    int row = it.get_global_id(2) + start_z;

                    /*!for values from x = HALF_LENGTH TO x= HALF_LENGTH +BOUND_LENGTH*/
                    int p_idx = depth * nz_nx + row * nx + column + start_x;
                    int p2_idx = depth * nz_nx + row * nx + boundary_length + start_x;
                    property_array[p_idx] = property_array[p2_idx];

                    /*!for values from x = nx-HALF_LENGTH TO x =
                     * nx-HALF_LENGTH-BOUND_LENGTH*/
                    p_idx = depth * nz_nx + row * nx + (end_x - 1 - column);
                    p2_idx = depth * nz_nx + row * nx + (end_x - 1 - boundary_length);
                    property_array[p_idx] = property_array[p2_idx];
                });
    });

    if (this->mUseTop) {
        /*!putting the nearest property_array adjacent to the boundary as the value
         * for all velocities at the boundaries for z and with all x and y */
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            auto global_range =
                    range<3>(end_x - start_x, end_y - start_y, boundary_length);
            auto local_range = range<3>(1, 1, 1);
            auto global_nd_range = nd_range<3>(global_range, local_range);

            cgh.parallel_for<class Homogenous_velocity_extension_Z>(
                    global_nd_range, [=](nd_item<3> it) {
                        int column = it.get_global_id(0) + start_x;
                        int depth = it.get_global_id(1) + start_y;
                        int row = it.get_global_id(2);

                        /*!for values from z = HALF_LENGTH TO z = HALF_LENGTH +BOUND_LENGTH */
                        int p_idx = depth * nz_nx + (start_z + row) * nx + column;
                        int p2_idx =
                                depth * nz_nx + (start_z + boundary_length) * nx + column;
                        property_array[p_idx] = property_array[p2_idx];

                        /*!for values from z = nz-HALF_LENGTH TO z =
                         * nz-HALF_LENGTH-BOUND_LENGTH*/
                        p_idx = depth * nz_nx + (end_z - 1 - row) * nx + column;
                        p2_idx = depth * nz_nx + (end_z - 1 - boundary_length) * nx + column;
                        property_array[p_idx] = property_array[p2_idx];
                    });
        });
    }
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
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            auto global_range =
                    range<3>(end_x - start_x, end_y - start_y, boundary_length);
            auto local_range = range<3>(1, 1, 1);
            auto global_nd_range = nd_range<3>(global_range, local_range);

            cgh.parallel_for<class Homogenous_top_extension>(
                    global_nd_range, [=](nd_item<3> it) {
                        int column = it.get_global_id(0) + start_x;
                        int depth = it.get_global_id(1) + start_y;
                        int row = it.get_global_id(2);

                        /*!for values from z = HALF_LENGTH TO z = HALF_LENGTH +BOUND_LENGTH */
                        int p_idx = depth * nz_nx + (start_z + row) * nx + column;
                        int p2_idx =
                                depth * nz_nx + (start_z + boundary_length) * nx + column;
                        property_array[p_idx] = property_array[p2_idx];
                    });
        });
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
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            auto global_range =
                    range<3>(end_x - start_x, end_y - start_y, boundary_length);
            auto local_range = range<3>(1, 1, 1);
            auto global_nd_range = nd_range<3>(global_range, local_range);

            cgh.parallel_for<class Homogenous_top_remover>(
                    global_nd_range, [=](nd_item<3> it) {
                        int column = it.get_global_id(0) + start_x;
                        int depth = it.get_global_id(1) + start_y;
                        int row = it.get_global_id(2);

                        /*!for values from z = HALF_LENGTH TO z = HALF_LENGTH +BOUND_LENGTH */
                        int p_idx = depth * nz_nx + (start_z + row) * nx + column;
                        property_array[p_idx] = 0;
                    });
        });
    }
}
