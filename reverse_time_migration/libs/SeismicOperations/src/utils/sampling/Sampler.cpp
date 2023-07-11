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
// Created by ahmed-ayyad on 17/01/2021.
//

#include <operations/utils/sampling/Sampler.hpp>

#include <operations/utils/interpolation/Interpolator.hpp>

#include <string>

using namespace operations::utils::sampling;
using namespace operations::dataunits;
using namespace operations::common;
using namespace operations::utils::interpolation;


void Sampler::Resize(float *input, float *output,
                     GridSize *apInputGridBox, GridSize *apOutputGridBox,
                     ComputationParameters *apParameters) {

    int pre_x, pre_y, pre_z;
    int post_x, post_y, post_z;
    std::string name;

    pre_x = apInputGridBox->nx;
    pre_y = apInputGridBox->ny;
    pre_z = apInputGridBox->nz;

    post_x = apOutputGridBox->nx;
    post_y = apOutputGridBox->ny;
    post_z = apOutputGridBox->nz;
    if (pre_x != post_x || pre_z != post_z || pre_y != post_y) {
        Interpolator::InterpolateTrilinear(
                input,
                output,
                pre_x, pre_z, pre_y,
                post_x, post_z, post_y,
                apParameters->GetBoundaryLength(),
                apParameters->GetHalfLength());
    } else {
        memcpy(output, input, sizeof(float) * pre_x * pre_z * pre_y);
    }
}

void Sampler::CalculateAdaptiveCellDimensions(GridBox *apGridBox,
                                              ComputationParameters *apParameters,
                                              int minimum_velocity) {
    uint old_nx = apGridBox->GetInitialGridSize(X_AXIS);
    uint old_nz = apGridBox->GetInitialGridSize(Z_AXIS);
    uint old_ny = apGridBox->GetInitialGridSize(Y_AXIS);

    float old_dx = apGridBox->GetInitialCellDimensions(X_AXIS);
    float old_dz = apGridBox->GetInitialCellDimensions(Z_AXIS);
    float old_dy = apGridBox->GetInitialCellDimensions(Y_AXIS);

    float old_nx_m = old_nx * old_dx;
    float old_nz_m = old_nz * old_dz;
    float old_ny_m = old_ny * old_dy;

    float maximum_frequency, minimum_wavelength;
    maximum_frequency = apParameters->GetSourceFrequency();
    minimum_wavelength = minimum_velocity / maximum_frequency;
    int stencil_order = apParameters->GetHalfLength();

    float minimum_wavelength_points;

    switch (stencil_order) {
        case O_2:
            minimum_wavelength_points = 10;
            break;
        case O_4:
            minimum_wavelength_points = 5;
            break;
        case O_8:
            minimum_wavelength_points = 4;
            break;
        case O_12:
            minimum_wavelength_points = 3.5;
            break;
        case O_16:
            minimum_wavelength_points = 3.2;
            break;
        default:
            minimum_wavelength_points = 4;
    }

    float meters_per_cell = (minimum_wavelength / minimum_wavelength_points) * 0.9; // safety factor

    float new_dx = meters_per_cell;
    float new_dz = meters_per_cell;
    float new_dy = 1;
    if (old_ny > 1) {
        new_dy = meters_per_cell;
    }

    apGridBox->SetCellDimensions(X_AXIS, new_dx);
    apGridBox->SetCellDimensions(Y_AXIS, new_dy);
    apGridBox->SetCellDimensions(Z_AXIS, new_dz);

    int bound_length = apParameters->GetBoundaryLength();
    int half_length = apParameters->GetHalfLength();

    int constant_offset = 2 * bound_length + 2 * half_length;

    int old_domain_x = old_nx - constant_offset;
    int old_domain_z = old_nz - constant_offset;
    int old_domain_y = old_ny - constant_offset;

    int new_domain_x = (old_domain_x * old_dx) / new_dx;
    int new_domain_z = (old_domain_z * old_dz) / new_dz;
    int new_domain_y = (old_domain_y * old_dy) / new_dy;

    int new_nx = new_domain_x + constant_offset;
    int new_nz = new_domain_z + constant_offset;
    int new_ny = new_domain_y + constant_offset;

    if (old_ny == 1) {
        new_ny = 1;
    }
    apGridBox->SetLogicalGridSize(X_AXIS, new_nx);
    apGridBox->SetLogicalGridSize(Z_AXIS, new_nz);
    apGridBox->SetLogicalGridSize(Y_AXIS, new_ny);

    int old_window_nx = apGridBox->GetLogicalWindowSize(X_AXIS);
    int old_window_nz = apGridBox->GetLogicalWindowSize(Z_AXIS);
    int old_window_ny = apGridBox->GetLogicalWindowSize(Y_AXIS);

    int new_window_nx = ((old_window_nx - constant_offset) * old_dx) / new_dx + constant_offset;
    int new_window_nz = ((old_window_nz - constant_offset) * old_dz) / new_dz + constant_offset;
    int new_window_ny = old_window_ny;

    if (old_window_ny > 1) {
        new_window_ny = ((old_window_ny - constant_offset) * old_dy) / new_dy + constant_offset;
    }

    apGridBox->SetLogicalWindowSize(X_AXIS, new_window_nx);
    apGridBox->SetLogicalWindowSize(Z_AXIS, new_window_nz);
    apGridBox->SetLogicalWindowSize(Y_AXIS, new_window_ny);

    int new_right_window = apParameters->GetRightWindow() * old_dx / new_dx;
    int new_left_window = apParameters->GetLeftWindow() * old_dx / new_dx;

    int new_front_window = apParameters->GetFrontWindow() * old_dy / new_dy;
    int new_back_window = apParameters->GetBackWindow() * old_dy / new_dy;

    int new_depth_window = apParameters->GetDepthWindow() * old_dz / new_dz;

    apParameters->SetRightWindow(new_right_window);
    apParameters->SetLeftWindow(new_left_window);
    apParameters->SetFrontWindow(new_front_window);
    apParameters->SetBackWindow(new_back_window);
    apParameters->SetDepthWindow(new_depth_window);
}
