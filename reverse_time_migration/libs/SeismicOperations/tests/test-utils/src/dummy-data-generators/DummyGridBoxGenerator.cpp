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
// Created by zeyad-osama on 01/02/2021.
//

#include <operations/test-utils/dummy-data-generators/DummyGridBoxGenerator.hpp>

using namespace operations::dataunits;


GridBox *generate_grid_box_no_wind_2d();

GridBox *generate_grid_box_inc_wind_2d();

GridBox *operations::testutils::generate_grid_box(
        OP_TU_DIMS aDims, OP_TU_WIND aWindow) {
    GridBox *gb = nullptr;
    if (aDims == OP_TU_2D && aWindow == OP_TU_NO_WIND) {
        gb = generate_grid_box_no_wind_2d();
    } else if (aDims == OP_TU_2D && aWindow == OP_TU_INC_WIND) {
        gb = generate_grid_box_inc_wind_2d();
    }
    return gb;
}

GridBox *generate_grid_box_no_wind_2d() {
    /*
     * Variables initialization for grid box.
     */

    int hl = 4;

    int nx = 23;
    int ny = 1;
    int nz = 23;

    int wnx = nx;
    int wny = ny;
    int wnz = nz;

    int sx = 0;
    int sy = 0;
    int sz = 0;

    float dx = 6.25f;
    float dy = 0.0f;
    float dz = 6.25f;

    float dt = 0.00207987f;

    auto *grid_box = new GridBox();

    /*
     * Setting variables inside grid box.
     */

    grid_box->SetLogicalGridSize(X_AXIS, nx);
    grid_box->SetLogicalGridSize(Y_AXIS, ny);
    grid_box->SetLogicalGridSize(Z_AXIS, nz);

    grid_box->SetLogicalWindowSize(X_AXIS, wnx);
    grid_box->SetLogicalWindowSize(Y_AXIS, wny);
    grid_box->SetLogicalWindowSize(Z_AXIS, wnz);

    grid_box->SetActualGridSize(X_AXIS, nx);
    grid_box->SetActualGridSize(Y_AXIS, ny);
    grid_box->SetActualGridSize(Z_AXIS, nz);

    grid_box->SetInitialGridSize(X_AXIS, nx);
    grid_box->SetInitialGridSize(Y_AXIS, ny);
    grid_box->SetInitialGridSize(Z_AXIS, nz);

    grid_box->SetActualWindowSize(X_AXIS, wnx);
    grid_box->SetActualWindowSize(Y_AXIS, wny);
    grid_box->SetActualWindowSize(Z_AXIS, wnz);

    grid_box->SetComputationGridSize(X_AXIS, wnx - 2 * hl);
    grid_box->SetComputationGridSize(Y_AXIS, wny);
    grid_box->SetComputationGridSize(Z_AXIS, wnz - 2 * hl);


    grid_box->SetCellDimensions(X_AXIS, dx);
    grid_box->SetCellDimensions(Y_AXIS, dy);
    grid_box->SetCellDimensions(Z_AXIS, dz);

    grid_box->SetInitialCellDimensions(X_AXIS, dx);
    grid_box->SetInitialCellDimensions(Y_AXIS, dy);
    grid_box->SetInitialCellDimensions(Z_AXIS, dz);

    grid_box->SetReferencePoint(X_AXIS, sx);
    grid_box->SetReferencePoint(Y_AXIS, sy);
    grid_box->SetReferencePoint(Z_AXIS, sz);

    grid_box->SetDT(dt);

    return grid_box;
}

GridBox *generate_grid_box_inc_wind_2d() {
    /*
    * Variables initialization for grid box.
    */

    int hl = 4;

    int nx = 23;
    int ny = 1;
    int nz = 23;

    int wnx = 21;
    int wny = 1;
    int wnz = 21;

    int sx = 0;
    int sy = 0;
    int sz = 0;

    float dx = 6.25f;
    float dy = 0.0f;
    float dz = 6.25f;

    float dt = 0.00207987f;

    auto *grid_box = new GridBox();

    /*
     * Setting variables inside grid box.
     */

    grid_box->SetLogicalGridSize(X_AXIS, nx);
    grid_box->SetLogicalGridSize(Y_AXIS, ny);
    grid_box->SetLogicalGridSize(Z_AXIS, nz);

    grid_box->SetLogicalWindowSize(X_AXIS, wnx);
    grid_box->SetLogicalWindowSize(Y_AXIS, wny);
    grid_box->SetLogicalWindowSize(Z_AXIS, wnz);

    grid_box->SetActualGridSize(X_AXIS, nx);
    grid_box->SetActualGridSize(Y_AXIS, ny);
    grid_box->SetActualGridSize(Z_AXIS, nz);

    grid_box->SetInitialGridSize(X_AXIS, nx);
    grid_box->SetInitialGridSize(Y_AXIS, ny);
    grid_box->SetInitialGridSize(Z_AXIS, nz);

    grid_box->SetActualWindowSize(X_AXIS, wnx);
    grid_box->SetActualWindowSize(Y_AXIS, wny);
    grid_box->SetActualWindowSize(Z_AXIS, wnz);

    grid_box->SetComputationGridSize(X_AXIS, wnx - 2 * hl);
    grid_box->SetComputationGridSize(Y_AXIS, wny);
    grid_box->SetComputationGridSize(Z_AXIS, wnz - 2 * hl);


    grid_box->SetCellDimensions(X_AXIS, dx);
    grid_box->SetCellDimensions(Y_AXIS, dy);
    grid_box->SetCellDimensions(Z_AXIS, dz);

    grid_box->SetInitialCellDimensions(X_AXIS, dx);
    grid_box->SetInitialCellDimensions(Y_AXIS, dy);
    grid_box->SetInitialCellDimensions(Z_AXIS, dz);

    grid_box->SetReferencePoint(X_AXIS, sx);
    grid_box->SetReferencePoint(Y_AXIS, sy);
    grid_box->SetReferencePoint(Z_AXIS, sz);

    grid_box->SetDT(dt);

    return grid_box;
}
