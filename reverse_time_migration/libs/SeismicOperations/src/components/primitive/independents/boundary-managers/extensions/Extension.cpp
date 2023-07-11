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

#include "operations/components/independents/concrete/boundary-managers/extensions/Extension.hpp"

#include <memory-manager/MemoryManager.h>

using namespace std;
using namespace operations::components::addons;
using namespace operations::dataunits;

Extension::Extension() = default;

Extension::~Extension() = default;

void Extension::SetHalfLength(uint aHalfLength) {
    this->mHalfLength = aHalfLength;
}

void Extension::SetBoundaryLength(uint aBoundaryLength) {
    this->mBoundaryLength = aBoundaryLength;
}

void Extension::SetGridBox(GridBox *apGridBox) {
    this->mpGridBox = apGridBox;
}

void Extension::SetProperty(float *property, float *window_property) {
    this->mProperties = property;
    this->mpWindowProperties = window_property;
}

void Extension::ExtendProperty() {
    int nx = mpGridBox->GetActualGridSize(X_AXIS);
    int ny = mpGridBox->GetActualGridSize(Y_AXIS);
    int nz = mpGridBox->GetActualGridSize(Z_AXIS);

    /**
     * The nx , ny and nz includes the inner domain + BOUND_LENGTH + HALF_LENGTH in
     * all dimensions and we want to extend the velocities at boundaries only with
     * the HALF_LENGTH excluded
     */
    int start_x = mHalfLength;
    int start_y = mHalfLength;
    int start_z = mHalfLength;

    int end_x = mpGridBox->GetLogicalGridSize(X_AXIS) - mHalfLength;
    int end_y = mpGridBox->GetLogicalGridSize(Y_AXIS) - mHalfLength;
    int end_z = mpGridBox->GetLogicalGridSize(Z_AXIS) - mHalfLength;

    /**
     * Change the values of velocities at
     * boundaries (HALF_LENGTH excluded) to zeros.
     */
    VelocityExtensionHelper(this->mProperties,
                            start_x, start_y, start_z,
                            end_x, end_y, end_z,
                            nx, ny, nz,
                            mBoundaryLength);
}

void Extension::ReExtendProperty() {
    /**
     * Re-Extend the velocities in case of window model.
    */
    int nx = mpGridBox->GetActualGridSize(X_AXIS);
    int ny = mpGridBox->GetActualGridSize(Y_AXIS);
    int nz = mpGridBox->GetActualGridSize(Z_AXIS);

    /**
     * The window size is a struct containing the window nx, ny and nz
     * with the HALF_LENGTH and BOUND_LENGTH in all dimensions.
     */
    int wnx = mpGridBox->GetActualWindowSize(X_AXIS);
    int wny = mpGridBox->GetActualWindowSize(Y_AXIS);
    int wnz = mpGridBox->GetActualWindowSize(Z_AXIS);

    if (mProperties == mpWindowProperties) {
        /// No window model, no need to re-extend so return from function

        /**
         * The nx, ny and nz includes the inner domain + BOUND_LENGTH +HALF_LENGTH
         * in all dimensions and we want to extend the velocities at boundaries only
         * with the HALF_LENGTH excluded
         */
        int start_x = mHalfLength;
        int start_y = mHalfLength;
        int start_z = mHalfLength;

        int end_x = mpGridBox->GetLogicalGridSize(X_AXIS) - mHalfLength;
        int end_y = mpGridBox->GetLogicalGridSize(Y_AXIS) - mHalfLength;
        int end_z = mpGridBox->GetLogicalGridSize(Z_AXIS) - mHalfLength;

        /**
         * No window model, no need to re-extend.
         * Just re-extend the top boundary.
         */
        this->TopLayerExtensionHelper(this->mProperties,
                                      start_x, start_y, start_z,
                                      end_x, end_y, end_z,
                                      nx, ny, nz,
                                      mBoundaryLength);
        return;
    } else {
        /// Window model.

        /**
         * We want to work in velocities inside window but with the HALF_LENGTH
         * excluded in all dimensions to reach the bound_length so it is applied in
         * start points by adding HALF_LENGTH also at end by subtract HALF_LENGTH.
         */
        int start_x = mHalfLength;
        int start_y = mHalfLength;
        int start_z = mHalfLength;

        int end_x = mpGridBox->GetLogicalWindowSize(X_AXIS) - mHalfLength;
        int end_y = mpGridBox->GetLogicalWindowSize(Y_AXIS) - mHalfLength;
        int end_z = mpGridBox->GetLogicalWindowSize(Z_AXIS) - mHalfLength;

        /// Extend the velocities at boundaries by zeros
        this->VelocityExtensionHelper(this->mpWindowProperties,
                                      start_x, start_y, start_z,
                                      end_x, end_y, end_z,
                                      wnx, wny, wnz,
                                      mBoundaryLength);
    }
}

void Extension::AdjustPropertyForBackward() {
    int nx = mpGridBox->GetActualGridSize(X_AXIS);
    int ny = mpGridBox->GetActualGridSize(Y_AXIS);
    int nz = mpGridBox->GetActualGridSize(Z_AXIS);

    /**
     * The window size is a struct containing the window nx, ny, and nz
     * with the HALF_LENGTH and BOUND_LENGTH in all dimensions.
     */
    int wnx = mpGridBox->GetActualWindowSize(X_AXIS);
    int wny = mpGridBox->GetActualWindowSize(Y_AXIS);
    int wnz = mpGridBox->GetActualWindowSize(Z_AXIS);

    /**
     * We want to work in velocities inside window but with the HALF_LENGTH
     * excluded in all dimensions to reach the bound_length so it is applied in
     * start points by adding HALF_LENGTH also at end by subtract HALF_LENGTH.
     */
    int start_x = mHalfLength;
    int start_y = mHalfLength;
    int start_z = mHalfLength;

    int end_x = mpGridBox->GetLogicalWindowSize(X_AXIS) - mHalfLength;
    int end_y = mpGridBox->GetLogicalWindowSize(Y_AXIS) - mHalfLength;
    int end_z = mpGridBox->GetLogicalWindowSize(Z_AXIS) - mHalfLength;
    if (ny == 1) {
        end_y = 1;
        start_y = 0;
    }
    this->TopLayerRemoverHelper(this->mpWindowProperties,
                                start_x, start_y, start_z,
                                end_x, end_y, end_z,
                                wnx, wny, wnz,
                                mBoundaryLength);
}