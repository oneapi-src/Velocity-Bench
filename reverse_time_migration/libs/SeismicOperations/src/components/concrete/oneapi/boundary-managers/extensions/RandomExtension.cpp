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
#include <operations/backend/OneAPIBackend.hpp>

#include <algorithm>
#include <cstdlib>

using namespace sycl;
using namespace std;
using namespace operations::components;
using namespace operations::components::addons;
using namespace operations::dataunits;
using namespace operations::backend;

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
    OneAPIBackend::GetInstance()->GetDeviceQueue()->submit(
            [&](handler &cgh) { cgh.memcpy(array, temp_arr, sizeof(float) * size); });
    OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
    free(temp_arr);
}

void RandomExtension::VelocityExtensionHelper(
        float *property_array,
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
    float *random_data;
    random_data = (float *) sycl::malloc_device(
            sizeof(float) * random_size,
            OneAPIBackend::GetInstance()->GetDeviceQueue()->get_device(),
            OneAPIBackend::GetInstance()->GetDeviceQueue()->get_context());
    float *dev_max_velocity = (float *) sycl::malloc_device(
            sizeof(float) * 1,
            OneAPIBackend::GetInstance()->GetDeviceQueue()->get_device(),
            OneAPIBackend::GetInstance()->GetDeviceQueue()->get_context());
    // In case of 2D
    if (ny == 1) {
        end_y = 1;
        start_y = 0;
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            cgh.single_task([=]() {
                // Get maximum property_array value
                for (int row = start_z + boundary_length; row < end_z - boundary_length;
                     row++) {
                    for (int column = start_x + boundary_length;
                         column < end_x - boundary_length; column++) {
                        if (dev_max_velocity[0] < property_array[row * nx + column]) {
                            dev_max_velocity[0] = property_array[row * nx + column];
                        }
                    }
                }
            });
        });
        OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            cgh.memcpy(&max_velocity, dev_max_velocity, sizeof(float) * 1);
        });
        OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
    } else {
        // Get maximum property_array value.
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            cgh.single_task([=]() {
                // Get maximum property_array_value
                for (int depth = start_y + boundary_length;
                     depth < end_y - boundary_length; depth++) {
                    for (int row = start_z + boundary_length;
                         row < end_z - boundary_length; row++) {
                        for (int column = start_x + boundary_length;
                             column < end_x - boundary_length; column++) {
                            if (dev_max_velocity[0] <
                                property_array[depth * nz_nx + row * nx + column]) {
                                dev_max_velocity[0] =
                                        property_array[depth * nz_nx + row * nx + column];
                            }
                        }
                    }
                }
            });
        });
        OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            cgh.memcpy(&max_velocity, dev_max_velocity, sizeof(float) * 1);
        });
        OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
        // general case for 3D
        randomize(random_data, random_size);
        /*!putting random values for velocities at the boundaries for y and with all
         * x and z */
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            auto global_range =
                    range<3>(end_x - start_x, boundary_length, end_z - start_z);
            auto local_range = range<3>(1, 1, 1);
            auto global_nd_range = nd_range<3>(global_range, local_range);

            cgh.parallel_for(
                    global_nd_range, [=](nd_item<3> it) {
                        int column = it.get_global_id(0) + start_x;
                        int depth = it.get_global_id(1);
                        int row = it.get_global_id(2) + start_z;

                        /*! Create temporary value */
                        float temp =
                                random_data[it.get_global_linear_id()] *
                                ((float) (boundary_length - (depth)) / boundary_length) *
                                max_velocity;
                        /*!for values from x = HALF_LENGTH TO x= HALF_LENGTH +BOUND_LENGTH*/
                        int p_idx = (depth + start_y) * nz_nx + row * nx + column;
                        int p2_idx =
                                (boundary_length + start_y) * nz_nx + row * nx + column;
                        property_array[p_idx] = _abs(property_array[p2_idx] - temp);
                        /*! Create temporary value */
                        temp = random_data[random_size - 1 - it.get_global_linear_id()] *
                               ((float) (boundary_length - (depth)) / boundary_length) *
                               max_velocity;
                        /*!for values from x = nx-HALF_LENGTH TO x =
                         * nx-HALF_LENGTH-BOUND_LENGTH*/
                        p_idx = (end_y - 1 - depth) * nz_nx + row * nx + column;
                        p2_idx = (end_y - 1 - boundary_length) * nz_nx + row * nx + column;
                        property_array[p_idx] = _abs(property_array[p2_idx] - temp);
                    });
        });
        OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
    }
    randomize(random_data, random_size);
    /*!putting random values for velocities at the boundaries for X and with all Y
     * and Z */
    OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
        auto global_range =
                range<3>(boundary_length, end_y - start_y, end_z - start_z);
        auto local_range = range<3>(1, 1, 1);
        auto global_nd_range = nd_range<3>(global_range, local_range);

        cgh.parallel_for(
                global_nd_range, [=](nd_item<3> it) {
                    int column = it.get_global_id(0);
                    int depth = it.get_global_id(1) + start_y;
                    int row = it.get_global_id(2) + start_z;

                    /*! Create temporary value */
                    float temp = random_data[it.get_global_linear_id()] *
                                 ((float) (boundary_length - (column)) / boundary_length) *
                                 max_velocity;
                    /*!for values from x = HALF_LENGTH TO x= HALF_LENGTH +BOUND_LENGTH*/
                    int p_idx = depth * nz_nx + row * nx + column + start_x;
                    int p2_idx = depth * nz_nx + row * nx + boundary_length + start_x;
                    property_array[p_idx] = _abs(property_array[p2_idx] - temp);
                    /*! Create temporary value */
                    temp = random_data[random_size - 1 - it.get_global_linear_id()] *
                           ((float) (boundary_length - (column)) / boundary_length) *
                           max_velocity;
                    /*!for values from x = nx-HALF_LENGTH TO x =
                     * nx-HALF_LENGTH-BOUND_LENGTH*/
                    p_idx = depth * nz_nx + row * nx + (end_x - 1 - column);
                    p2_idx = depth * nz_nx + row * nx + (end_x - 1 - boundary_length);
                    property_array[p_idx] = _abs(property_array[p2_idx] - temp);
                });
    });
    OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();

    randomize(random_data, random_size);
    /*!putting random values for velocities at the boundaries for z and with all x
     * and y */
    OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
        auto global_range =
                range<3>(end_x - start_x, end_y - start_y, boundary_length);
        auto local_range = range<3>(1, 1, 1);
        auto global_nd_range = nd_range<3>(global_range, local_range);

        cgh.parallel_for(
                global_nd_range, [=](nd_item<3> it) {
                    int column = it.get_global_id(0) + start_x;
                    int depth = it.get_global_id(1) + start_y;
                    int row = it.get_global_id(2);

                    /*! Create temporary value */
                    float temp = random_data[it.get_global_linear_id()] *
                                 ((float) (boundary_length - (row)) / boundary_length) *
                                 max_velocity;
                    /*!for values from x = HALF_LENGTH TO x= HALF_LENGTH +BOUND_LENGTH*/
                    int p_idx = depth * nz_nx + (start_z + row) * nx + column;
                    int p2_idx =
                            depth * nz_nx + (start_z + boundary_length) * nx + column;
                    // Remove top layer boundary : give value as zero since having top
                    // layer random boundaries will introduce too much noise.
                    property_array[p_idx] = 0; //_abs(property_array[p2_idx] - temp);
                    /*! Create temporary value */
                    temp = random_data[random_size - 1 - it.get_global_linear_id()] *
                           ((float) (boundary_length - (row)) / boundary_length) *
                           max_velocity;
                    /*!for values from x = nx-HALF_LENGTH TO x =
                     * nx-HALF_LENGTH-BOUND_LENGTH*/
                    p_idx = depth * nz_nx + (end_z - 1 - row) * nx + column;
                    p2_idx = depth * nz_nx + (end_z - 1 - boundary_length) * nx + column;
                    property_array[p_idx] = _abs(property_array[p2_idx] - temp);
                });
    });

    OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();

    randomize(random_data, random_size);
    // Random-Corners in the boundaries nx-nz boundary intersection at bottom--
    // top boundaries not needed.
    OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
        auto global_range =
                range<3>(boundary_length, end_y - start_y, boundary_length);
        auto local_range = range<3>(1, 1, 1);
        auto global_nd_range = nd_range<3>(global_range, local_range);

        cgh.parallel_for(
                global_nd_range, [=](nd_item<3> it) {
                    int column = it.get_global_id(0);
                    int depth = it.get_global_id(1) + start_y;
                    int row = it.get_global_id(2);

                    uint offset = std::min(row, column);
                    /*!for values from z = HALF_LENGTH TO z = HALF_LENGTH +BOUND_LENGTH */
                    /*! and for x = HALF_LENGTH to x = HALF_LENGTH + BOUND_LENGTH */
                    /*! Top left boundary in other words */
                    property_array[depth * nz_nx + (start_z + row) * nx + column +
                                   start_x] = 0;
                    /*!for values from z = nz-HALF_LENGTH TO z =
                     * nz-HALF_LENGTH-BOUND_LENGTH*/
                    /*! and for x = HALF_LENGTH to x = HALF_LENGTH + BOUND_LENGTH */
                    /*! Bottom left boundary in other words */
                    float temp = random_data[it.get_global_linear_id()] *
                                 ((float) (boundary_length - (offset)) / boundary_length) *
                                 max_velocity;
                    int p_idx = depth * nz_nx + (end_z - 1 - row) * nx + column + start_x;
                    int p2_idx = depth * nz_nx + (end_z - 1 - boundary_length) * nx +
                                 start_x + boundary_length;
                    property_array[p_idx] = _abs(property_array[p2_idx] - temp);
                    /*!for values from z = HALF_LENGTH TO z = HALF_LENGTH +BOUND_LENGTH */
                    /*! and for x = nx-HALF_LENGTH to x = nx-HALF_LENGTH - BOUND_LENGTH */
                    /*! Top right boundary in other words */
                    property_array[depth * nz_nx + (start_z + row) * nx +
                                   (end_x - 1 - column)] = 0;
                    /*!for values from z = nz-HALF_LENGTH TO z =
                     * nz-HALF_LENGTH-BOUND_LENGTH*/
                    /*! and for x = nx-HALF_LENGTH to x = nx - HALF_LENGTH - BOUND_LENGTH
                     */
                    /*! Bottom right boundary in other words */
                    temp = random_data[random_size - 1 - it.get_global_linear_id()] *
                           ((float) (boundary_length - (offset)) / boundary_length) *
                           max_velocity;
                    /*!for values from x = nx-HALF_LENGTH TO x =
                     * nx-HALF_LENGTH-BOUND_LENGTH*/
                    p_idx = depth * nz_nx + (end_z - 1 - row) * nx + (end_x - 1 - column);
                    p2_idx = depth * nz_nx + (end_z - 1 - boundary_length) * nx +
                             (end_x - 1 - boundary_length);
                    property_array[p_idx] = _abs(property_array[p2_idx] - temp);
                });
    });

    OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();

    randomize(random_data, random_size);
    // If 3-D, zero corners in the y-x and y-z plans.
    if (ny > 1) {
        // Random-Corners in the boundaries ny-nz boundary intersection at bottom--
        // top boundaries not needed.
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            auto global_range =
                    range<3>(end_x - start_x, boundary_length, boundary_length);
            auto local_range = range<3>(1, 1, 1);
            auto global_nd_range = nd_range<3>(global_range, local_range);

            cgh.parallel_for(
                    global_nd_range, [=](nd_item<3> it) {
                        int column = it.get_global_id(0) + start_x;
                        int depth = it.get_global_id(1);
                        int row = it.get_global_id(2);

                        uint offset = std::min(row, depth);
                        /*!for values from z = HALF_LENGTH TO z = HALF_LENGTH +BOUND_LENGTH
                         */
                        /*! and for y = HALF_LENGTH to y = HALF_LENGTH + BOUND_LENGTH */
                        property_array[(depth + start_y) * nz_nx + (start_z + row) * nx +
                                       column] = 0;
                        /*!for values from z = nz-HALF_LENGTH TO z =
                         * nz-HALF_LENGTH-BOUND_LENGTH*/
                        /*! and for y = HALF_LENGTH to y = HALF_LENGTH + BOUND_LENGTH */
                        float temp =
                                random_data[it.get_global_linear_id()] *
                                ((float) (boundary_length - (offset)) / boundary_length) *
                                max_velocity;
                        int p_idx =
                                (depth + start_y) * nz_nx + (end_z - 1 - row) * nx + column;
                        int p2_idx = (start_y + boundary_length) * nz_nx +
                                     (end_z - 1 - boundary_length) * nx + column;
                        property_array[p_idx] = _abs(property_array[p2_idx] - temp);
                        /*!for values from z = HALF_LENGTH TO z = HALF_LENGTH +BOUND_LENGTH
                         */
                        /*! and for y = ny-HALF_LENGTH to y = ny-HALF_LENGTH - BOUND_LENGTH
                         */
                        property_array[(end_y - 1 - depth) * nz_nx + (start_z + row) * nx +
                                       column] = 0;
                        /*!for values from z = nz-HALF_LENGTH TO z =
                         * nz-HALF_LENGTH-BOUND_LENGTH */
                        /*! and for y = ny-HALF_LENGTH to y = ny - HALF_LENGTH -
                         * BOUND_LENGTH */
                        temp = random_data[random_size - 1 - it.get_global_linear_id()] *
                               ((float) (boundary_length - (offset)) / boundary_length) *
                               max_velocity;
                        /*!for values from x = nx-HALF_LENGTH TO x =
                         * nx-HALF_LENGTH-BOUND_LENGTH*/
                        p_idx =
                                (end_y - 1 - depth) * nz_nx + (end_z - 1 - row) * nx + column;
                        p2_idx = (end_y - 1 - boundary_length) * nz_nx +
                                 (end_z - 1 - boundary_length) * nx + column;
                        property_array[p_idx] = _abs(property_array[p2_idx] - temp);
                    });
        });

        OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();

        randomize(random_data, random_size);
        // Zero-Corners in the boundaries nx-ny boundary intersection on the top
        // layer--boundaries not needed.
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            auto global_range =
                    range<3>(boundary_length, boundary_length, boundary_length);
            auto local_range = range<3>(1, 1, 1);
            auto global_nd_range = nd_range<3>(global_range, local_range);

            cgh.parallel_for(
                    global_nd_range, [=](nd_item<3> it) {
                        int column = it.get_global_id(0);
                        int depth = it.get_global_id(1);
                        int row = it.get_global_id(2) + start_z;

                        /*!for values from y = HALF_LENGTH TO y = HALF_LENGTH +BOUND_LENGTH
                         */
                        /*! and for x = HALF_LENGTH to x = HALF_LENGTH + BOUND_LENGTH */
                        property_array[(depth + start_y) * nz_nx + row * nx + column +
                                       start_x] = 0;
                        /*!for values from y = ny-HALF_LENGTH TO y =
                         * ny-HALF_LENGTH-BOUND_LENGTH*/
                        /*! and for x = HALF_LENGTH to x = HALF_LENGTH + BOUND_LENGTH */
                        property_array[(end_y - 1 - depth) * nz_nx + row * nx + column +
                                       start_x] = 0;
                        /*!for values from y = HALF_LENGTH TO y = HALF_LENGTH +BOUND_LENGTH
                         */
                        /*! and for x = nx-HALF_LENGTH to x = nx-HALF_LENGTH - BOUND_LENGTH
                         */
                        property_array[(depth + start_y) * nz_nx + row * nx +
                                       (end_x - 1 - column)] = 0;
                        /*!for values from y = ny-HALF_LENGTH TO y =
                         * ny-HALF_LENGTH-BOUND_LENGTH*/
                        /*! and for x = nx-HALF_LENGTH to x = nx - HALF_LENGTH -
                         * BOUND_LENGTH */
                        property_array[(end_y - 1 - depth) * nz_nx + row * nx +
                                       (end_x - 1 - column)] = 0;
                    });
        });

        OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();

        randomize(random_data, random_size);
        // Random-Corners in the boundaries nx-ny boundary intersection.
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            auto global_range =
                    range<3>(boundary_length, boundary_length, boundary_length);
            auto local_range = range<3>(1, 1, 1);
            auto global_nd_range = nd_range<3>(global_range, local_range);

            cgh.parallel_for(
                    global_nd_range, [=](nd_item<3> it) {
                        int column = it.get_global_id(0);
                        int depth = it.get_global_id(1);
                        int row = it.get_global_id(2) + start_z;

                        uint offset = std::min(column, depth);
                        /*!for values from y = HALF_LENGTH TO y = HALF_LENGTH +BOUND_LENGTH
                         */
                        /*! and for x = HALF_LENGTH to x = HALF_LENGTH + BOUND_LENGTH */
                        float temp =
                                random_data[it.get_global_linear_id()] *
                                ((float) (boundary_length - (offset)) / boundary_length) *
                                max_velocity;
                        property_array[(depth + start_y) * nz_nx + row * nx + column +
                                       start_x] =
                                _abs(property_array[(boundary_length + start_y) * nz_nx +
                                                    row * nx + boundary_length + start_x] -
                                     temp);
                        /*!for values from y = ny-HALF_LENGTH TO y =
                         * ny-HALF_LENGTH-BOUND_LENGTH*/
                        /*! and for x = HALF_LENGTH to x = HALF_LENGTH + BOUND_LENGTH */
                        temp = random_data[random_size - 1 - it.get_global_linear_id()] *
                               ((float) (boundary_length - (offset)) / boundary_length) *
                               max_velocity;
                        property_array[(end_y - 1 - depth) * nz_nx + row * nx + column +
                                       start_x] =
                                _abs(property_array[(end_y - 1 - boundary_length) * nz_nx +
                                                    row * nx + boundary_length + start_x] -
                                     temp);
                        /*!for values from y = HALF_LENGTH TO y = HALF_LENGTH +BOUND_LENGTH
                         */
                        /*! and for x = nx-HALF_LENGTH to x = nx-HALF_LENGTH - BOUND_LENGTH
                         */
                        temp = (random_data[it.get_global_linear_id()] *
                                random_data[random_size - 1 - it.get_global_linear_id()]) *
                               ((float) (boundary_length - (offset)) / boundary_length) *
                               max_velocity;
                        property_array[(depth + start_y) * nz_nx + row * nx +
                                       (end_x - 1 - column)] =
                                _abs(property_array[(boundary_length + start_y) * nz_nx +
                                                    row * nx + (end_x - 1 - boundary_length)] -
                                     temp);
                        /*!for values from y = ny-HALF_LENGTH TO y =
                         * ny-HALF_LENGTH-BOUND_LENGTH*/
                        /*! and for x = nx-HALF_LENGTH to x = nx - HALF_LENGTH -
                         * BOUND_LENGTH */
                        temp = (random_data[it.get_global_linear_id()] +
                                random_data[random_size - 1 - it.get_global_linear_id()]) *
                               ((float) (boundary_length - (offset)) / boundary_length) *
                               max_velocity;
                        property_array[(end_y - 1 - depth) * nz_nx + row * nx +
                                       (end_x - 1 - column)] =
                                _abs(property_array[(end_y - 1 - boundary_length) * nz_nx +
                                                    row * nx + (end_x - 1 - boundary_length)] -
                                     temp);
                    });
        });
        OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
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
