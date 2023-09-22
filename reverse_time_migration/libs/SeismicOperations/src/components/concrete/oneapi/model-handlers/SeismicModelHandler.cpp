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
// Created by zeyad-osama on 28/07/2020.
//

#include <operations/components/independents/concrete/model-handlers/SeismicModelHandler.hpp>

#include <operations/backend/OneAPIBackend.hpp>

#define make_divisible(v, d) (v + (d - (v % d)))

using namespace sycl;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;
using namespace operations::backend;

void SeismicModelHandler::PreprocessModel() {
  int nx = mpGridBox->GetActualGridSize(X_AXIS);
  int ny = mpGridBox->GetActualGridSize(Y_AXIS);
  int nz = mpGridBox->GetActualGridSize(Z_AXIS);

  float dt2 = mpGridBox->GetDT() * mpGridBox->GetDT();

  // Preprocess the velocity model by calculating the dt2 * c2 component of
  // the wave equation.
  OneAPIBackend::GetInstance()->GetDeviceQueue()->submit(
      [&](sycl::handler &cgh) {
        auto global_range = range<2>(nx, nz);
        auto local_range = range<2>(1, 1);
        auto global_nd_range = nd_range<2>(global_range, local_range);
        float *vel_device = mpGridBox->Get(PARM | GB_VEL)->GetNativePointer();

        cgh.parallel_for(
            global_nd_range, [=](sycl::nd_item<2> it) {
              int x = it.get_global_id(0);
              int z = it.get_global_id(1);
              float value = vel_device[z * nx + x];
              vel_device[z * nx + x] =
                  value * value * dt2;
            });
      });
  /////OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
}

void SeismicModelHandler::SetupWindow() {
  if (mpParameters->IsUsingWindow()) {
    uint wnx = mpGridBox->GetActualWindowSize(X_AXIS);
    uint wnz = mpGridBox->GetActualWindowSize(Z_AXIS);
    uint wny = mpGridBox->GetActualWindowSize(Y_AXIS);
    uint nx = mpGridBox->GetActualGridSize(X_AXIS);
    uint nz = mpGridBox->GetActualGridSize(Z_AXIS);
    uint ny = mpGridBox->GetActualGridSize(Y_AXIS);
    uint sx = mpGridBox->GetWindowStart(X_AXIS);
    uint sz = mpGridBox->GetWindowStart(Z_AXIS);
    uint sy = mpGridBox->GetWindowStart(Y_AXIS);
    uint offset = mpParameters->GetHalfLength() + mpParameters->GetBoundaryLength();
    uint start_x = offset;
    uint end_x = mpGridBox->GetLogicalWindowSize(X_AXIS) - offset;
    uint start_z = offset;
    uint end_z = mpGridBox->GetLogicalWindowSize(Z_AXIS) - offset;
    uint start_y = 0;
    uint end_y = 1;
    if (ny != 1) {
      start_y = offset;
      end_y = mpGridBox->GetLogicalWindowSize(Y_AXIS) - offset;
    }
    OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](sycl::handler &cgh) {
      auto global_range = range<2>(end_x - start_x, end_z - start_z);
      auto local_range = range<2>(1, 1);
      auto global_nd_range = nd_range<2>(global_range, local_range);
      float *vel = mpGridBox->Get(PARM | GB_VEL)->GetNativePointer();
      float *w_vel = mpGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer();
      cgh.parallel_for(global_nd_range, [=](sycl::nd_item<2> it) {
        int x = it.get_global_id(0) + start_x;
        int z = it.get_global_id(1) + start_z;
        uint offset_window = z * wnx + x;
        uint offset_full = (sy) * nx * nz + (z + sz) * nx + x + sx;
        w_vel[offset_window] = vel[offset_full];
      });
    });
    /////OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
  }
}

void SeismicModelHandler::SetupPadding() {
  auto grid = mpGridBox;
  auto parameters = mpParameters;
  uint block_x = parameters->GetBlockX();
  uint block_z = parameters->GetBlockZ();
  uint nx = grid->GetLogicalWindowSize(X_AXIS);
  uint nz = grid->GetLogicalWindowSize(Z_AXIS);
  uint inx = nx - 2 * parameters->GetHalfLength();
  uint inz = nz - 2 * parameters->GetHalfLength();
  // Store old values of nx,nz,ny to use in boundaries/etc....

  this->mpGridBox->SetActualGridSize(X_AXIS, this->mpGridBox->GetLogicalGridSize(X_AXIS));
  this->mpGridBox->SetActualGridSize(Y_AXIS, this->mpGridBox->GetLogicalGridSize(Y_AXIS));
  this->mpGridBox->SetActualGridSize(Z_AXIS, this->mpGridBox->GetLogicalGridSize(Z_AXIS));
  this->mpGridBox->SetActualWindowSize(X_AXIS, this->mpGridBox->GetLogicalWindowSize(X_AXIS));
  this->mpGridBox->SetActualWindowSize(Y_AXIS, this->mpGridBox->GetLogicalWindowSize(Y_AXIS));
  this->mpGridBox->SetActualWindowSize(Z_AXIS, this->mpGridBox->GetLogicalWindowSize(Z_AXIS));

  if (block_x > inx) {
    block_x = inx;
    std::cout << "Block Factor x > domain size... Reduced to domain size"
              << std::endl;
  }
  if (block_z > inz) {
    block_z = inz;
    std::cout << "Block Factor z > domain size... Reduced to domain size"
              << std::endl;
  }
  if (block_x % 16 != 0 && block_x != 1) {
    block_x = make_divisible(
        block_x,
        16); // Ensure block in x is divisible by 16(biggest vector length).
    std::cout << "Adjusting block factor in x to make it divisible by "
                 "16(Possible Vector Length)..."
              << std::endl;
  }
  if (inx % block_x != 0) {
    std::cout
        << "Adding padding to make domain divisible by block size in  x-axis"
        << std::endl;
    inx = make_divisible(inx, block_x);
    grid->SetComputationGridSize(X_AXIS, inx);
    nx = inx + 2 * parameters->GetHalfLength();
  }
  if (inz % block_z != 0) {
    std::cout
        << "Adding padding to make domain divisible by block size in  z-axis"
        << std::endl;
    inz = make_divisible(inz, block_z);
    nz = inz + 2 * parameters->GetHalfLength();
  }
  if (nx % 16 != 0) {
    std::cout << "Adding padding to ensure alignment of each row" << std::endl;
    nx = make_divisible(nx, 16);
  }
  // Set grid with the padded values.
  grid->SetActualWindowSize(X_AXIS, nx);
  grid->SetActualWindowSize(Z_AXIS, nz);
  parameters->SetBlockX(block_x);
  parameters->SetBlockZ(block_z);
  if (!parameters->IsUsingWindow()) {
    grid->SetActualGridSize(X_AXIS, grid->GetActualWindowSize(X_AXIS));
    grid->SetActualGridSize(Z_AXIS, grid->GetActualWindowSize(Z_AXIS));
  }

  this->mpGridBox->SetComputationGridSize(Z_AXIS,
                                          this->mpGridBox->GetActualWindowSize(Z_AXIS) -
                                          2 * this->mpParameters->GetHalfLength());
  if (this->mpGridBox->GetLogicalWindowSize(Y_AXIS) > 1) {
    this->mpGridBox->SetComputationGridSize(Y_AXIS,
                                            this->mpGridBox->GetActualWindowSize(Y_AXIS) -
                                            2 * this->mpParameters->GetHalfLength());
  } else {
    this->mpGridBox->SetComputationGridSize(Y_AXIS, 1);
  }
}
