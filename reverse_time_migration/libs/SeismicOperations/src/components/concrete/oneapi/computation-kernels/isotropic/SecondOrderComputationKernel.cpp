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
// Created by amr-nasr on 11/21/19.
//

#include <operations/components/independents/concrete/computation-kernels/isotropic/SecondOrderComputationKernel.hpp>

#include <operations/backend/OneAPIBackend.hpp>

#include <timer/Timer.h>
#include <memory-manager/MemoryManager.h>

#include <cstring>
#include <cassert>

#define fma(a, b, c) (a) * (b) + (c)

using namespace sycl;
using namespace std;
using namespace operations::components;
using namespace operations::dataunits;
using namespace operations::common;
using namespace operations::backend;

template void SecondOrderComputationKernel::Compute<true, O_2>();

template void SecondOrderComputationKernel::Compute<true, O_4>();

template void SecondOrderComputationKernel::Compute<true, O_8>();

template void SecondOrderComputationKernel::Compute<true, O_12>();

template void SecondOrderComputationKernel::Compute<true, O_16>();

template void SecondOrderComputationKernel::Compute<false, O_2>();

template void SecondOrderComputationKernel::Compute<false, O_4>();

template void SecondOrderComputationKernel::Compute<false, O_8>();

template void SecondOrderComputationKernel::Compute<false, O_12>();

template void SecondOrderComputationKernel::Compute<false, O_16>();

template<bool IS_2D_, HALF_LENGTH HALF_LENGTH_>
void SecondOrderComputationKernel::Compute() {
    // Read parameters into local variables to be shared.

    size_t nx = mpGridBox->GetActualWindowSize(X_AXIS);
    size_t nz = mpGridBox->GetActualWindowSize(Z_AXIS);

    float *prev_base = mpGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();
    float *curr_base = mpGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    float *next_base = mpGridBox->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();

    float *vel_base = mpGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer();

    // Pre-compute the coefficients for each direction.
    int hl = HALF_LENGTH_;
    if (OneAPIBackend::GetInstance()->GetAlgorithm() == SYCL_ALGORITHM::CPU) {
        ////std::cout << "CPU NOT SUPPORTED" << std::endl;
        ////std::cout << "RUNNING CPU!!!" << std::endl;
        /////assert(0);
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            const float *current = curr_base;
            float *next = next_base;
            const float *prev = prev_base;
            const float *vel = vel_base;
            const float *c_x = mpCoeffX->GetNativePointer();
            const float *c_z = mpCoeffZ->GetNativePointer();
            const float c_xyz = mCoeffXYZ;
            const int *v = mpVerticalIdx->GetNativePointer();
            const size_t wnx = nx;
            const size_t wnz = nz;
            auto global_range = range<2>(mpGridBox->GetComputationGridSize(X_AXIS),
                                         mpGridBox->GetComputationGridSize(Z_AXIS));
            auto local_range = range<2>(mpParameters->GetBlockX(), mpParameters->GetBlockZ());
            auto global_offset = id<2>(HALF_LENGTH_, HALF_LENGTH_);
            auto global_nd_range = nd_range<2>(global_range, local_range, global_offset);


            cgh.parallel_for(
                    global_nd_range, [=](nd_item<2> it) {
                        int x = it.get_global_id(0);
                        int z = it.get_global_id(1);

                        int idx = wnx * z + x;


                        float value = current[idx] * c_xyz;

                        value = fma(current[idx - 1] + current[idx + 1], c_x[0], value);
                        value = fma(current[idx - v[0]] + current[idx + v[0]], c_z[0], value);

                        if (HALF_LENGTH_ > 1) {
                            value = fma(current[idx - 2] + current[idx + 2], c_x[1], value);
                            value = fma(current[idx - v[1]] + current[idx + v[1]], c_z[1], value);
                        }
                        if (HALF_LENGTH_ > 2) {
                            value = fma(current[idx - 3] + current[idx + 3], c_x[2], value);
                            value = fma(current[idx - 4] + current[idx + 4], c_x[3], value);
                            value = fma(current[idx - v[2]] + current[idx + v[2]], c_z[2], value);
                            value = fma(current[idx - v[3]] + current[idx + v[3]], c_z[3], value);
                        }
                        if (HALF_LENGTH_ > 4) {
                            value = fma(current[idx - 5] + current[idx + 5], c_x[4], value);
                            value = fma(current[idx - 6] + current[idx + 6], c_x[5], value);
                            value = fma(current[idx - v[4]] + current[idx + v[4]], c_z[4], value);
                            value = fma(current[idx - v[5]] + current[idx + v[5]], c_z[5], value);
                        }
                        if (HALF_LENGTH_ > 6) {
                            value = fma(current[idx - 7] + current[idx + 7], c_x[6], value);
                            value = fma(current[idx - 8] + current[idx + 8], c_x[7], value);
                            value = fma(current[idx - v[6]] + current[idx + v[6]], c_z[6], value);
                            value = fma(current[idx - v[7]] + current[idx + v[7]], c_z[7], value);
                        }

                        next[idx] = (2 * current[idx]) - prev[idx] + (vel[idx] * value);

                    });
        });
    } else if (OneAPIBackend::GetInstance()->GetAlgorithm() == SYCL_ALGORITHM::GPU_SHARED) {
        std::cout << "GPU SHARED NOT SUPPORTED" << std::endl;
        assert(0);

//////        int compute_nz = mpGridBox->GetComputationGridSize(Z_AXIS);
//////        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
//////            auto global_range = range<2>(compute_nz, mpGridBox->GetComputationGridSize(X_AXIS));
//////            auto local_range = range<2>(mpParameters->GetBlockZ(), mpParameters->GetBlockX());
//////            sycl::nd_range<2> workgroup_range(global_range, local_range);
//////            const float *current = curr_base;
//////            const float *prev = prev_base;
//////            float *next = next_base;
//////            const float *vel = vel_base;
//////            const float *c_x = mpCoeffX->GetNativePointer();
//////            const float *c_z = mpCoeffZ->GetNativePointer();
//////            const float c_xyz = mCoeffXYZ;
//////            const int *v = mpVerticalIdx->GetNativePointer();
//////            const int idx_range = mpParameters->GetBlockZ();
//////            const int local_nx = mpParameters->GetBlockX() + 2 * hl;
//////            auto localRange_ptr_cur = range<1>(((mpParameters->GetBlockX() + (2 * hl)) *
//////                                                (mpParameters->GetBlockZ() + (2 * hl))));
//////            //  Create an accessor for SLM buffer
//////            accessor<float, 1, access::mode::read_write, access::target::local> tab(
//////                    localRange_ptr_cur, cgh);
//////            cgh.parallel_for<class secondOrderComputation_dpcpp>(
//////                    workgroup_range, [=](nd_item<2> it) {
//////                        float *local = tab.get_pointer();
//////                        int idx =
//////                                it.get_global_id(1) + hl + (it.get_global_id(0) + hl) * nx;
//////                        size_t id0 = it.get_local_id(1);
//////                        size_t id1 = it.get_local_id(0);
//////                        size_t identifiant = (id0 + hl) + (id1 + hl) * local_nx;
//////                        float c_x_loc[HALF_LENGTH_];
//////                        float c_z_loc[HALF_LENGTH_];
//////                        int v_loc[HALF_LENGTH_];
//////                        // Set local coeff.
//////                        for (unsigned int iter = 0; iter < HALF_LENGTH_; iter++) {
//////                            c_x_loc[iter] = c_x[iter];
//////                            c_z_loc[iter] = c_z[iter];
//////                            v_loc[iter] = (iter + 1) * local_nx;
//////                        }
//////                        bool copyHaloX = false;
//////                        bool copyHaloY = false;
//////                        const unsigned int items_X = it.get_local_range(1);
//////                        // Set Shared Memory.
//////                        local[identifiant] = current[idx];
//////                        if (id0 < HALF_LENGTH_) {
//////                            copyHaloX = true;
//////                        }
//////                        if (id1 < HALF_LENGTH_) {
//////                            copyHaloY = true;
//////                        }
//////                        if (copyHaloX) {
//////                            local[identifiant - HALF_LENGTH_] = current[idx - HALF_LENGTH_];
//////                            local[identifiant + items_X] = current[idx + items_X];
//////                        }
//////                        if (copyHaloY) {
//////                            local[identifiant - HALF_LENGTH_ * local_nx] =
//////                                    current[idx - HALF_LENGTH_ * nx];
//////                            local[identifiant + idx_range * local_nx] =
//////                                    current[idx + idx_range * nx];
//////                        }
//////                        it.barrier(access::fence_space::local_space);
//////                        float value = 0;
//////                        value = fma(local[identifiant], c_xyz, value);
//////                        for (int iter = 1; iter <= HALF_LENGTH_; iter++) {
//////                            value = fma(local[identifiant - iter], c_x_loc[iter - 1], value);
//////                            value = fma(local[identifiant + iter], c_x_loc[iter - 1], value);
//////                        }
//////                        for (int iter = 1; iter <= HALF_LENGTH_; iter++) {
//////                            value = fma(local[identifiant - v_loc[iter - 1]],
//////                                        c_z_loc[iter - 1], value);
//////                            value = fma(local[identifiant + v_loc[iter - 1]],
//////                                        c_z_loc[iter - 1], value);
//////                        }
//////                        value = fma(vel[idx], value, -prev[idx]);
//////                        value = fma(2.0f, local[identifiant], value);
//////                        next[idx] = value;
//////                    });
//////        });
    } else if (OneAPIBackend::GetInstance()->GetAlgorithm() == SYCL_ALGORITHM::GPU_SEMI_SHARED) {
        std::cout << "GPU SEMI SHARED SUPPORTED" << std::endl;
        assert(0);

//////        int compute_nz = mpGridBox->GetComputationGridSize(Z_AXIS) / mpParameters->GetBlockZ();
//////        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
//////            auto global_range = range<2>(compute_nz, mpGridBox->GetComputationGridSize(X_AXIS));
//////            auto local_range = range<2>(1, mpParameters->GetBlockX());
//////            sycl::nd_range<2> workgroup_range(global_range, local_range);
//////            const float *current = curr_base;
//////            const float *prev = prev_base;
//////            float *next = next_base;
//////            const float *vel = vel_base;
//////            const float *c_x = mpCoeffX->GetNativePointer();
//////            const float *c_z = mpCoeffZ->GetNativePointer();
//////            const float c_xyz = mCoeffXYZ;
//////            const int *v = mpVerticalIdx->GetNativePointer();
//////            const int idx_range = mpParameters->GetBlockZ();
//////            const int local_nx = mpParameters->GetBlockX() + 2 * hl;
//////            const int local_nz = mpParameters->GetBlockZ() + 2 * hl;
//////            auto localRange_ptr_cur = range<1>(((mpParameters->GetBlockX() + (2 * hl)) *
//////                                                (mpParameters->GetBlockZ() + (2 * hl))));
//////            //  Create an accessor for SLM buffer
//////            accessor<float, 1, access::mode::read_write, access::target::local> tab(
//////                    localRange_ptr_cur, cgh);
//////            cgh.parallel_for<class secondOrderComputation_dpcpp>(
//////                    workgroup_range, [=](nd_item<2> it) {
//////                        float *local = tab.get_pointer();
//////                        int idx = it.get_global_id(1) + hl +
//////                                  (it.get_global_id(0) * idx_range + hl) * nx;
//////                        size_t id0 = it.get_local_id(1);
//////                        size_t identifiant = (id0 + hl) + hl * local_nx;
//////                        float c_x_loc[HALF_LENGTH_];
//////                        float c_z_loc[HALF_LENGTH_];
//////                        int v_loc[HALF_LENGTH_];
//////                        // Set local coeff.
//////                        for (unsigned int iter = 0; iter < HALF_LENGTH_; iter++) {
//////                            c_x_loc[iter] = c_x[iter];
//////                            c_z_loc[iter] = c_z[iter];
//////                            v_loc[iter] = (iter + 1) * local_nx;
//////                        }
//////                        bool copyHaloX = false;
//////                        if (id0 < HALF_LENGTH_)
//////                            copyHaloX = true;
//////                        const unsigned int items_X = it.get_local_range(1);
//////                        int load_identifiant = identifiant - hl * local_nx;
//////                        int load_idx = idx - hl * nx;
//////                        // Set Shared Memory.
//////                        for (int i = 0; i < local_nz; i++) {
//////                            local[load_identifiant] = current[load_idx];
//////                            if (copyHaloX) {
//////                                local[load_identifiant - HALF_LENGTH_] =
//////                                        current[load_idx - HALF_LENGTH_];
//////                                local[load_identifiant + items_X] = current[load_idx + items_X];
//////                            }
//////                            load_idx += nx;
//////                            load_identifiant += local_nx;
//////                        }
//////                        it.barrier(access::fence_space::local_space);
//////                        for (int i = 0; i < idx_range; i++) {
//////                            float value = 0;
//////                            value = fma(local[identifiant], c_xyz, value);
//////                            for (int iter = 1; iter <= HALF_LENGTH_; iter++) {
//////                                value =
//////                                        fma(local[identifiant - iter], c_x_loc[iter - 1], value);
//////                                value =
//////                                        fma(local[identifiant + iter], c_x_loc[iter - 1], value);
//////                            }
//////                            for (int iter = 1; iter <= HALF_LENGTH_; iter++) {
//////                                value = fma(local[identifiant - v_loc[iter - 1]],
//////                                            c_z_loc[iter - 1], value);
//////                                value = fma(local[identifiant + v_loc[iter - 1]],
//////                                            c_z_loc[iter - 1], value);
//////                            }
//////                            value = fma(vel[idx], value, -prev[idx]);
//////                            value = fma(2.0f, local[identifiant], value);
//////                            next[idx] = value;
//////                            idx += nx;
//////                            identifiant += local_nx;
//////                        }
//////                    });
//////        });
    } else if (OneAPIBackend::GetInstance()->GetAlgorithm() == SYCL_ALGORITHM::GPU) {
        int compute_nz = mpGridBox->GetComputationGridSize(Z_AXIS) / mpParameters->GetBlockZ();
        OneAPIBackend::GetInstance()->GetDeviceQueue()->submit([&](handler &cgh) {
            auto global_range = range<2>(compute_nz, mpGridBox->GetComputationGridSize(X_AXIS));
            auto local_range = range<2>(1, mpParameters->GetBlockX());
            sycl::nd_range<2> workgroup_range(global_range, local_range);
            const float *current = curr_base;
            const float *prev = prev_base;
            float *next = next_base;
            const float *vel = vel_base;
            const float *c_x = mpCoeffX->GetNativePointer();
            const float *c_z = mpCoeffZ->GetNativePointer();
            const float c_xyz = mCoeffXYZ;
            const int *v = mpVerticalIdx->GetNativePointer();
            const int idx_range = mpParameters->GetBlockZ();
            const int pad = 0;
            auto localRange_ptr_cur =
                    range<1>((mpParameters->GetBlockX() + (2 * hl) + pad));
            //  Create an accessor for SLM buffer
            accessor<float, 1, access::mode::read_write, access::target::local> tab(
                    localRange_ptr_cur, cgh);
            cgh.parallel_for(
                    workgroup_range, [=](nd_item<2> it) {
                        float *local = tab.get_pointer();
                        int idx = it.get_global_id(1) + hl +
                                  (it.get_global_id(0) * idx_range + hl) * nx;
                        size_t id0 = it.get_local_id(1);
                        size_t identifiant = (id0 + hl);
                        float c_x_loc[HALF_LENGTH_];
                        float c_z_loc[HALF_LENGTH_];
                        int v_end = v[HALF_LENGTH_ - 1];
                        float front[HALF_LENGTH_ + 1];
                        float back[HALF_LENGTH_];
                        for (unsigned int iter = 0; iter <= HALF_LENGTH_; iter++) {
                            front[iter] = current[idx + nx * iter];
                        }
                        for (unsigned int iter = 1; iter <= HALF_LENGTH_; iter++) {
                            back[iter - 1] = current[idx - nx * iter];
                            c_x_loc[iter - 1] = c_x[iter - 1];
                            c_z_loc[iter - 1] = c_z[iter - 1];
                        }
                        bool copyHaloX = false;
                        if (id0 < HALF_LENGTH_)
                            copyHaloX = true;
                        const unsigned int items_X = it.get_local_range(1);
                        for (int i = 0; i < idx_range; i++) {
                            local[identifiant] = front[0];
                            if (copyHaloX) {
                                local[identifiant - HALF_LENGTH_] = current[idx - HALF_LENGTH_];
                                local[identifiant + items_X] = current[idx + items_X];
                            }
                            it.barrier(access::fence_space::local_space); /// [JT>>:] This barrier has no effect on performance improvement/degradation
                            float value = 0;
                            value = fma(local[identifiant], c_xyz, value);
                            for (int iter = 1; iter <= HALF_LENGTH_; iter++) {
                                value =
                                        fma(local[identifiant - iter], c_x_loc[iter - 1], value);
                                value =
                                        fma(local[identifiant + iter], c_x_loc[iter - 1], value);
                            }
                            for (int iter = 1; iter <= HALF_LENGTH_; iter++) {
                                value = fma(front[iter], c_z_loc[iter - 1], value);
                                value = fma(back[iter - 1], c_z_loc[iter - 1], value);
                            }
                            value = fma(vel[idx], value, -prev[idx]);
                            value = fma(2.0f, local[identifiant], value);
                            next[idx] = value;
                            idx += nx;
                            for (unsigned int iter = HALF_LENGTH_ - 1; iter > 0; iter--) {
                                back[iter] = back[iter - 1];
                            }
                            back[0] = front[0];
                            for (unsigned int iter = 0; iter < HALF_LENGTH_; iter++) {
                                front[iter] = front[iter + 1];
                            }
                            // Only one new data-point read from global memory
                            // in z-dimension (depth)
                            front[HALF_LENGTH_] = current[idx + v_end];
                        }
                    });
        });
    }
    ////OneAPIBackend::GetInstance()->GetDeviceQueue()->wait();
}
