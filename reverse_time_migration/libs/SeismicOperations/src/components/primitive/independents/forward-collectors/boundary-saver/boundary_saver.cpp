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
// Created by mirna-moawad on 1/20/20.
//

#include <operations/components/independents/concrete/forward-collectors/boundary-saver/boundary_saver.h>

using namespace operations::components::helpers;
using namespace operations::common;
using namespace operations::dataunits;


void operations::components::helpers::save_boundaries(
        GridBox *apGridBox, ComputationParameters *apParameters,
        float *backup_boundaries, uint step, uint boundary_size) {
    uint index = 0;
    uint size_of_boundaries = boundary_size;
    uint time_step = step;
    uint half_length = apParameters->GetHalfLength();
    uint bound_length = apParameters->GetBoundaryLength();
    uint offset = half_length + bound_length;

    uint start_y = 0;
    uint end_y = 1;

    uint ny = apGridBox->GetActualGridSize(Y_AXIS);

    uint wnx = apGridBox->GetActualWindowSize(X_AXIS);
    uint wny = apGridBox->GetActualWindowSize(Y_AXIS);
    uint wnz = apGridBox->GetActualWindowSize(Z_AXIS);

    uint start_z = offset;
    uint end_z = apGridBox->GetLogicalWindowSize(Z_AXIS) - offset;
    uint start_x = offset;
    uint end_x = apGridBox->GetLogicalWindowSize(X_AXIS) - offset;
    uint wnznx = wnx * wnz;

    if (ny > 1) {
        start_y = offset;
        end_y = apGridBox->GetLogicalWindowSize(Y_AXIS) - offset;
    }

    float *current_pressure = apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();

    for (int iy = start_y; iy < end_y; iy++) {
        for (int iz = start_z; iz < end_z; iz++) {
            for (int ix = 0; ix < half_length; ix++) {
                backup_boundaries[time_step * size_of_boundaries + index] =
                        current_pressure[iy * wnznx + iz * wnx + bound_length + ix];
                index++;
                backup_boundaries[time_step * size_of_boundaries + index] =
                        current_pressure[iy * wnznx + iz * wnx + (wnx - bound_length - 1) - ix];
                index++;
            }
        }
    }
    for (int iy = start_y; iy < end_y; iy++) {
        for (int iz = 0; iz < half_length; iz++) {
            for (int ix = start_x; ix < end_x; ix++) {
                backup_boundaries[time_step * size_of_boundaries + index] =
                        current_pressure[iy * wnznx + (bound_length + iz) * wnx + ix];
                index++;
                backup_boundaries[time_step * size_of_boundaries + index] =
                        current_pressure[iy * wnznx + (wnz - bound_length - 1 - iz) * wnx + ix];
                index++;
            }
        }
    }
    if (ny > 1) {
        for (int iy = 0; iy < half_length; iy++) {
            for (int iz = start_z; iz < end_z; iz++) {
                for (int ix = start_x; ix < end_x; ix++) {
                    backup_boundaries[time_step * size_of_boundaries + index] =
                            current_pressure[(bound_length + iy) * wnznx + iz * wnx + ix];
                    index++;
                    backup_boundaries[time_step * size_of_boundaries + index] =
                            current_pressure[(wny - bound_length - 1 - iy) * wnznx + iz * wnx + ix];
                    index++;
                }
            }
        }
    }
}

void operations::components::helpers::restore_boundaries(
        GridBox *apMainGrid, GridBox *apInternalGrid,
        ComputationParameters *apParameters,
        const float *backup_boundaries, uint step, uint boundary_size) {
    uint index = 0;
    uint size_of_boundaries = boundary_size;
    uint time_step = step;
    uint half_length = apParameters->GetHalfLength();
    uint bound_length = apParameters->GetBoundaryLength();
    uint offset = half_length + bound_length;

    uint start_y = 0;
    uint end_y = 1;

    uint ny = apMainGrid->GetActualGridSize(Y_AXIS);

    uint wnx = apMainGrid->GetActualWindowSize(X_AXIS);
    uint wny = apMainGrid->GetActualWindowSize(Y_AXIS);
    uint wnz = apMainGrid->GetActualWindowSize(Z_AXIS);

    uint window_size = wnx * wny * wnz;

    uint start_z = offset;
    uint end_z = apMainGrid->GetLogicalWindowSize(Z_AXIS) - offset;
    uint start_x = offset;
    uint end_x = apMainGrid->GetLogicalWindowSize(X_AXIS) - offset;
    uint wnznx = wnx * wnz;
    if (ny > 1) {
        start_y = offset;
        end_y = apMainGrid->GetLogicalWindowSize(Y_AXIS) - offset;
    }

    float *current_pressure = apInternalGrid->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();

    for (int iy = start_y; iy < end_y; iy++) {
        for (int iz = start_z; iz < end_z; iz++) {
            for (int ix = 0; ix < half_length; ix++) {
                current_pressure[iy * wnznx + iz * wnx + bound_length + ix] =
                        backup_boundaries[time_step * size_of_boundaries + index];
                index++;
                current_pressure[iy * wnznx + iz * wnx + (wnx - bound_length - 1) - ix] =
                        backup_boundaries[time_step * size_of_boundaries + index];
                index++;
            }
        }
    }
    for (int iy = start_y; iy < end_y; iy++) {
        for (int iz = 0; iz < half_length; iz++) {
            for (int ix = start_x; ix < end_x; ix++) {
                current_pressure[iy * wnznx + (bound_length + iz) * wnx + ix] =
                        backup_boundaries[time_step * size_of_boundaries + index];
                index++;
                current_pressure[iy * wnznx + (wnz - bound_length - 1 - iz) * wnx + ix] =
                        backup_boundaries[time_step * size_of_boundaries + index];
                index++;
            }
        }
    }
    if (ny > 1) {
        for (int iy = 0; iy < half_length; iy++) {
            for (int iz = start_z; iz < end_z; iz++) {
                for (int ix = start_x; ix < end_x; ix++) {
                    current_pressure[(bound_length + iy) * wnznx + iz * wnx + ix] =
                            backup_boundaries[time_step * size_of_boundaries + index];
                    index++;
                    current_pressure[(wny - bound_length - 1 - iy) * wnznx + iz * wnx + ix] =
                            backup_boundaries[time_step * size_of_boundaries + index];
                    index++;
                }
            }
        }
    }

    // Reflect changes that occurred to host buffer on native buffer
    apInternalGrid->Get(WAVE | GB_PRSS | CURR | DIR_Z)->ReflectOnNative();
}
