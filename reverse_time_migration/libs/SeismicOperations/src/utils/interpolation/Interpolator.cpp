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
// Created by zeyad-osama on 08/07/2020.
//

#include <operations/utils/interpolation/Interpolator.hpp>

#include <memory-manager/MemoryManager.h>

#include <iostream>

using namespace operations::utils::interpolation;
using namespace operations::dataunits;


float lerp1(float t, float a, float b);

float lerp2(float *t_xy, const float *vertices);

float lerp3(float *t_xyz, const float *vertices);

void interpolate_2D(const float *old_grid,
                    float *new_grid,
                    int old_nx, int old_nz,
                    int new_nx, int new_nz,
                    int bound_length,
                    int half_length);

void interpolate_3D(const float *old_grid,
                    float *new_grid,
                    int old_nx, int old_nz, int old_ny,
                    int new_nx, int new_nz, int new_ny,
                    int bound_length,
                    int half_length);

float *
Interpolator::Interpolate(TracesHolder *apTraceHolder, uint actual_nt, float total_time, INTERPOLATION aInterpolation) {
    if (aInterpolation == SPLINE) {
        Interpolator::InterpolateLinear(apTraceHolder, actual_nt, total_time);
    }
    return nullptr;
}

float *Interpolator::InterpolateLinear(TracesHolder *apTraceHolder, uint actual_nt, float total_time) {
    uint sample_nt = apTraceHolder->SampleNT;
    if (actual_nt <= sample_nt) {
        std::cerr << "Interpolation terminated..." << std::endl;
        std::cerr << "Actual size should be at least equal sample size..." << std::endl;
        return nullptr;
    }

    auto *interpolated_trace = (float *) mem_allocate(
            sizeof(float), actual_nt * apTraceHolder->TraceSizePerTimeStep,
            "interpolated-traces");

    double step = ((float) sample_nt) / actual_nt;
    double t_curr, t_next, t_intr, trace_curr, trace_next, slope, fx;

    uint trace_size_per_timestep = apTraceHolder->TraceSizePerTimeStep;
    for (int i = 0; i < trace_size_per_timestep; ++i) {
        int idx = 0;
        for (int t = 0; t < sample_nt - 1; ++t) {
            t_curr = t;
            t_next = t + 1;
            trace_curr = apTraceHolder->Traces[t * trace_size_per_timestep + i];
            trace_next = apTraceHolder->Traces[(t + 1) * trace_size_per_timestep + i];
            slope = (trace_next - trace_curr) / (t_next - t_curr);

            while (true) {
                t_intr = idx * step;
                if (t_intr > t_next || idx == actual_nt - 1) {
                    break;
                }
                fx = trace_curr + (slope * (t_intr - t_curr));
                interpolated_trace[idx++ * trace_size_per_timestep + i] = fx;
            }
        }
    }

    mem_free(apTraceHolder->Traces);
    apTraceHolder->Traces = interpolated_trace;
    apTraceHolder->SampleNT = actual_nt;
    apTraceHolder->SampleDT = total_time / actual_nt;

    return interpolated_trace;
}

void Interpolator::InterpolateTrilinear(float *old_grid, float *new_grid,
                                        int old_nx, int old_nz, int old_ny,
                                        int new_nx, int new_nz, int new_ny,
                                        int bound_length,
                                        int half_length) {
    if (old_ny > 1) {
        interpolate_3D(old_grid, new_grid,
                       old_nx, old_nz, old_ny,
                       new_nx, new_nz, new_ny,
                       bound_length,
                       half_length);
    } else {
        interpolate_2D(old_grid, new_grid,
                       old_nx, old_nz,
                       new_nx, new_nz,
                       bound_length,
                       half_length);
    }
}

inline float lerp1(float t, float a, float b) {
    if (t < 0) return a;
    if (t > 1) return b;
    return (a * (1 - t)) + (b * t);
}

float lerp2(float *t_xy, const float *vertices) {

    float *v_00_10, *v_01_11, *results_lerp1;
    float result;
    v_00_10 = (float *) malloc(2 * sizeof(float));
    v_01_11 = (float *) malloc(2 * sizeof(float));
    results_lerp1 = (float *) malloc(2 * sizeof(float));

    for (int i = 0; i < 2; i++) {
        v_00_10[i] = vertices[i];
        v_01_11[i] = vertices[i + 2];
    }

    results_lerp1[0] = lerp1(t_xy[0], v_00_10[0], v_00_10[1]);
    results_lerp1[1] = lerp1(t_xy[0], v_01_11[0], v_01_11[1]);

    result = lerp1(t_xy[1], results_lerp1[0], results_lerp1[1]);

    free(v_00_10);
    free(v_01_11);
    free(results_lerp1);

    return result;
}

float lerp3(float *t_xyz, const float *vertices) {

    float *v_000_100_010_110, *v_001_101_011_111, *results_lerp2, *t_xy;
    float result;

    v_000_100_010_110 = (float *) malloc(4 * sizeof(float));
    v_001_101_011_111 = (float *) malloc(4 * sizeof(float));
    t_xy = (float *) malloc(2 * sizeof(float));
    results_lerp2 = (float *) malloc(2 * sizeof(float));

    for (int i = 0; i < 4; i++) {
        v_000_100_010_110[i] = vertices[i];
        v_001_101_011_111[i] = vertices[i + 4];
    }

    t_xy[0] = t_xyz[0];
    t_xy[1] = t_xyz[1];

    results_lerp2[0] = lerp2(t_xy, v_000_100_010_110);
    results_lerp2[1] = lerp2(t_xy, v_001_101_011_111);

    result = lerp1(t_xyz[2], results_lerp2[0], results_lerp2[1]);

    free(v_000_100_010_110);
    free(v_001_101_011_111);
    free(results_lerp2);
    free(t_xy);

    return result;
}

void interpolate_2D(const float *old_grid,
                    float *new_grid,
                    int old_nx, int old_nz,
                    int new_nx, int new_nz,
                    int bound_length,
                    int half_length) {

    int old_domain_size_x = old_nx - (2 * bound_length) - (2 * half_length);
    int old_domain_size_z = old_nz - (2 * bound_length) - (2 * half_length);

    int new_domain_size_x = new_nx - (2 * bound_length) - (2 * half_length);
    int new_domain_size_z = new_nz - (2 * bound_length) - (2 * half_length);

    auto *vertices = (float *) malloc(4 * sizeof(float));
    auto *t_xy = (float *) malloc(2 * sizeof(float));

    float old_grid_ix_float;
    float old_grid_iz_float;

    int old_grid_ix_int;
    int old_grid_iz_int;

    int offsets[4][2] = {{0, 0},
                         {1, 0},
                         {0, 1},
                         {1, 1}};

    int idx;
    for (int iz = 0; iz < new_domain_size_z; iz++) {
        for (int ix = 0; ix < new_domain_size_x; ix++) {
            idx = ((iz + (bound_length + half_length)) * new_nx) + ix + (bound_length + half_length);

            old_grid_ix_float = (float) ix * (old_domain_size_x - 1) / (new_domain_size_x);
            old_grid_iz_float = (float) iz * (old_domain_size_z - 1) / (new_domain_size_z);

            old_grid_ix_int = (int) old_grid_ix_float + (bound_length + half_length);
            old_grid_iz_int = (int) old_grid_iz_float + (bound_length + half_length);

            t_xy[0] = old_grid_ix_float - old_grid_ix_int;
            t_xy[1] = old_grid_iz_float - old_grid_iz_int;

            int og_idx;
            for (int i = 0; i < 4; i++) {
                og_idx = ((old_grid_iz_int + offsets[i][1]) * old_nx) +
                         (old_grid_ix_int + offsets[i][0]);
                vertices[i] = old_grid[og_idx];
            }
            new_grid[idx] = lerp2(t_xy, vertices);
        }
    }
    free(vertices);
    free(t_xy);
}

void interpolate_3D(const float *old_grid,
                    float *new_grid,
                    int old_nx, int old_nz, int old_ny,
                    int new_nx, int new_nz, int new_ny,
                    int bound_length,
                    int half_length) {
    int old_domain_size_x = old_nx - (2 * bound_length) - (2 * half_length);
    int old_domain_size_z = old_nz - (2 * bound_length) - (2 * half_length);
    int old_domain_size_y = old_ny - (2 * bound_length) - (2 * half_length);

    int new_domain_size_x = new_nx - (2 * bound_length) - (2 * half_length);
    int new_domain_size_z = new_nz - (2 * bound_length) - (2 * half_length);
    int new_domain_size_y = new_ny - (2 * bound_length) - (2 * half_length);

    auto *vertices = (float *) malloc(8 * sizeof(float));
    auto *t_xyz = (float *) malloc(3 * sizeof(float));

    float old_grid_ix_float;
    float old_grid_iy_float;
    float old_grid_iz_float;

    int old_grid_ix_int;
    int old_grid_iz_int;
    int old_grid_iy_int;

    int offsets[8][3] = {{0, 0, 0},
                         {1, 0, 0},
                         {0, 1, 0},
                         {1, 1, 0},
                         {0, 0, 1},
                         {1, 0, 1},
                         {0, 1, 1},
                         {1, 1, 1}};

    int idx;
    for (int iy = 0; iy < new_domain_size_y; iy++) {
        for (int iz = 0; iz < new_domain_size_z; iz++) {
            for (int ix = 0; ix < new_domain_size_x; ix++) {
                idx = ((iy + bound_length + half_length) * new_nz * new_nx)
                      + ((iz + (bound_length + half_length)) * new_nx)
                      + ix + (bound_length + half_length);

                old_grid_ix_float = (float) ix * (old_domain_size_x - 1) / (new_domain_size_x);
                old_grid_iz_float = (float) iz * (old_domain_size_z - 1) / (new_domain_size_z);
                old_grid_iy_float = (float) iy * (old_domain_size_y - 1) / (new_domain_size_y);

                old_grid_ix_int = (int) old_grid_ix_float + (bound_length + half_length);
                old_grid_iz_int = (int) old_grid_iz_float + (bound_length + half_length);
                old_grid_iy_int = (int) old_grid_iy_float + (bound_length + half_length);

                t_xyz[0] = old_grid_ix_float - old_grid_ix_int;
                t_xyz[1] = old_grid_iz_float - old_grid_iz_int;
                t_xyz[2] = old_grid_iy_float - old_grid_iy_int;

                int og_idx;
                for (int i = 0; i < 8; i++) {
                    og_idx = ((old_grid_iy_int + offsets[i][2]) * old_nz * old_nx)
                             + ((old_grid_iz_int + offsets[i][1]) * old_nx)
                             + (old_grid_ix_int + offsets[i][0]);

                    vertices[i] = old_grid[og_idx];
                }
                new_grid[idx] = lerp3(t_xyz, vertices);
            }
        }
    }
    free(vertices);
    free(t_xyz);
}