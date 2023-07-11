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
// Created by zeyad-osama on 26/09/2020.
//

#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>

#include <operations/backend/OneAPIBackend.hpp>

using namespace sycl;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;
using namespace operations::backend;


void WaveFieldsMemoryHandler::FirstTouch(float *ptr, GridBox *apGridBox, bool enable_window) {
  if (OneAPIBackend::GetInstance()->GetAlgorithm() == SYCL_ALGORITHM::CPU) {
    /////std::cout << "CPU algorithm not supported" << std::endl;
    /////assert(0);
    int nx, nz, compute_nx, compute_nz;

    if (enable_window) {
      nx = apGridBox->GetActualWindowSize(X_AXIS);
      nz = apGridBox->GetActualWindowSize(Z_AXIS);
      compute_nx = apGridBox->GetComputationGridSize(X_AXIS);
      compute_nz = apGridBox->GetComputationGridSize(Z_AXIS);
    } else {
      nx = apGridBox->GetActualGridSize(X_AXIS);
      nz = apGridBox->GetActualGridSize(Z_AXIS);
      compute_nx = (nx - 2 * mpParameters->GetHalfLength());
      compute_nz = (nz - 2 * mpParameters->GetHalfLength());
    }
    OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
      auto global_range = range<2>(compute_nx, compute_nz);
      auto local_range = range<2>(mpParameters->GetBlockX(), mpParameters->GetBlockZ());
      auto starting_offset = id<2>(mpParameters->GetHalfLength(), mpParameters->GetHalfLength());
      auto global_nd_range = nd_range<2>(global_range, local_range, starting_offset);
      float *curr_base = ptr;

      cgh.parallel_for<class first_touch>(
          global_nd_range, [=](nd_item<2> it) {
            int idx = it.get_global_id(1) * nx + it.get_global_id(0);
            curr_base[idx] = 0;
          });
    });
    OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
    Device::MemSet(ptr, 0, nx * nz * sizeof(float));
  }
}
