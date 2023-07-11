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
// Created by ingy-mounir on 02/02/2021.
//

#include <operations/components/independents/concrete/forward-collectors/ReversePropagation.hpp>

#include <operations/common/DataTypes.h>
#include <operations/components/dependency/concrete/HasDependents.hpp>
#include <operations/components/independents/concrete/computation-kernels/isotropic/SecondOrderComputationKernel.hpp>
#include <operations/configurations/concrete/JSONConfigurationMap.hpp>
#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>
#include <operations/test-utils/dummy-data-generators/DummyConfigurationMapGenerator.hpp>
#include <operations/test-utils/dummy-data-generators/DummyGridBoxGenerator.hpp>
#include <operations/test-utils/dummy-data-generators/DummyParametersGenerator.hpp>
#include <operations/test-utils/NumberHelpers.hpp>
#include <operations/test-utils/EnvironmentHandler.hpp>

#include <libraries/catch/catch.hpp>

using namespace std;
using namespace operations;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;
using namespace operations::configuration;
using namespace operations::testutils;
using namespace operations::helpers;


/**
 * @note
 * 1. FetchForward();
 *      1.0 apply computation step case  no injection
 *
 * 2. SaveForward():
 *      1.0 do nothing if there is no injection enabled
 *      - check that it internal time step is not increased (this->mTimeStep)
 *      i.e. can't do this, it is internal
 *
 * 3. void ResetGrid(bool is_forward_run);
 *      2 cases:
 *          2.0 reset wave field with zeros
 *          2.1 forward : nothing more than resetting and then injection if enabled
 *          2.2 backward  : it swaps pointers , if no wave fields it allocates .
 * 4. dataunits::GridBox *GetForwardGrid();
 *      1 case:
 *          1.0 check getting internal forward grid : means dims and current pressure
 */
void TEST_CASE_FORWARD_COLLECTOR_REVERSE_NO_INJECTION(GridBox *apGridBox,
                                                      ComputationParameters *apParameters,
                                                      ConfigurationMap *apConfigurationMap) {
    /*
     * Environment setting (i.e. Backend setting initialization).
     */
    set_environment();

    /*
     * Register and allocate parameters and wave fields in 
     * grid box according to the current test case.
     */

    auto pressure_curr = new FrameBuffer<float>();
    auto pressure_prev = new FrameBuffer<float>();
    auto velocity = new FrameBuffer<float>();

    auto aMemoryHandler = new WaveFieldsMemoryHandler(apConfigurationMap);

    float nt = 5;
    apGridBox->SetNT(nt);

    /*
     * Variables initialization for grid box.
     */

    int nx, ny, nz;
    int wnx, wnz, wny;

    nx = apGridBox->GetActualGridSize(X_AXIS);
    ny = apGridBox->GetActualGridSize(Y_AXIS);
    nz = apGridBox->GetActualGridSize(Z_AXIS);

    wnx = apGridBox->GetActualWindowSize(X_AXIS);
    wny = apGridBox->GetActualWindowSize(Y_AXIS);
    wnz = apGridBox->GetActualWindowSize(Z_AXIS);

    uint window_size = wnx * wny * wnz;
    uint size = nx * ny * nz;

    pressure_curr->Allocate(window_size);
    pressure_prev->Allocate(window_size);
    velocity->Allocate(size);

    apGridBox->RegisterWaveField(WAVE | GB_PRSS | CURR | DIR_Z, pressure_curr);
    apGridBox->RegisterWaveField(WAVE | GB_PRSS | PREV | DIR_Z, pressure_prev);
    apGridBox->RegisterWaveField(WAVE | GB_PRSS | NEXT | DIR_Z, pressure_prev);
    apGridBox->RegisterParameter(PARM | GB_VEL, velocity);

    float temp_vel[size];
    float dt = apGridBox->GetDT();

    for (int iy = 0; iy < ny; iy++) {
        for (int iz = 0; iz < nz; iz++) {
            for (int ix = 0; ix < nx; ix++) {
                temp_vel[iz * nx + ix + (iy * nx * nz)] = 1500;
                temp_vel[iz * nx + ix + (iy * nx * nz)] *=
                        temp_vel[iz * nx + ix + (iy * nx * nz)] * dt * dt;
            }
        }
    }
    Device::MemSet(pressure_curr->GetNativePointer(), 0.0f, window_size * sizeof(float));
    Device::MemSet(pressure_prev->GetNativePointer(), 0.0f, window_size * sizeof(float));
    Device::MemCpy(velocity->GetNativePointer(), temp_vel, size * sizeof(float), Device::COPY_HOST_TO_DEVICE);

    aMemoryHandler->SetComputationParameters(apParameters);

    auto computation_kernel = new SecondOrderComputationKernel(apConfigurationMap);
    computation_kernel->SetGridBox(apGridBox);
    computation_kernel->SetComputationParameters(apParameters);


    auto dependent_components_map = new ComponentsMap<DependentComponent>();
    dependent_components_map->Set(MEMORY_HANDLER, aMemoryHandler);

    auto components_map = new ComponentsMap<Component>();
    components_map->Set(COMPUTATION_KERNEL, computation_kernel);

    auto forward_collector = new ReversePropagation(apConfigurationMap);
    forward_collector->SetComputationParameters(apParameters);
    forward_collector->SetDependentComponents(dependent_components_map);
    forward_collector->SetComponentsMap(components_map);
    forward_collector->SetGridBox(apGridBox);
    forward_collector->AcquireConfiguration();

    /*
     * Test for rest grid case true , means the wva fields should be zero 
     */

    Device::MemSet(pressure_curr->GetNativePointer(), 1.0f, window_size * sizeof(float));
    Device::MemSet(pressure_prev->GetNativePointer(), 1.0f, window_size * sizeof(float));

    forward_collector->ResetGrid(true);
    auto curr_pressure = apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    auto prev_pressure = apGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetHostPointer();

    int misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (curr_pressure[index] != 0) misses += 1;

            }
        }
    }
    REQUIRE(misses == 0);

    misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (prev_pressure[index] != 0) misses += 1;

            }
        }
    }
    REQUIRE(misses == 0);

    /*  
     * Test for get forward grid  in false option
     * and case of reset grid false , it return the internal grd
     */

    forward_collector->ResetGrid(false);
    auto grid_box = forward_collector->GetForwardGrid();

    REQUIRE(grid_box->GetActualGridSize(X_AXIS) == nx);
    REQUIRE(grid_box->GetActualGridSize(Y_AXIS) == ny);
    REQUIRE(grid_box->GetActualGridSize(Z_AXIS) == nz);
    REQUIRE(grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(grid_box->Get(PARM | GB_VEL)->GetNativePointer() != nullptr);

    auto pres_curr = grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto pres_prev = grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();

    /*
    * Check for swapping pointers after applying reset grid with false option
    */

    forward_collector->ResetGrid(false);

    auto swap_grid_box = forward_collector->GetForwardGrid();

    auto swap_pres_curr = swap_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto swap_pres_prev = swap_grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();

    REQUIRE(swap_pres_curr == pres_prev);
    REQUIRE(swap_pres_prev == pres_curr);

    /*
     * Testing fetch forward.
     * A step function is applied
     */

    auto fetch_grid_box = forward_collector->GetForwardGrid();

    auto h_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    int location = (wnx / 2) + (wnz / 2) * wnx + (wny / 2) * wnx * wnz;
    h_pressure[location] = 1;

    auto d_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    Device::MemCpy(d_pressure, h_pressure, window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);

    for (int it = 0; it < int(nt); it++) {
        forward_collector->FetchForward();
    }

    auto fetch_pres = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();

    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (fetch_pres[index] != 0) misses += 1;

            }
        }
    }
    REQUIRE(misses > 0);


    delete apGridBox;
    delete apParameters;
    delete apConfigurationMap;

    delete components_map;
    delete dependent_components_map;
    delete computation_kernel;
    delete forward_collector;
}

void TEST_CASE_FORWARD_COLLECTOR_REVERSE_INC_INJECTION(GridBox *apGridBox,
                                                       ComputationParameters *apParameters,
                                                       ConfigurationMap *apConfigurationMap) {
    /*
     * Environment setting (i.e. Backend setting initialization).
     */
    set_environment();

    /*
     * Register and allocate parameters and wave fields in
     * grid box according to the current test case.
     */

    auto pressure_curr = new FrameBuffer<float>();
    auto pressure_prev = new FrameBuffer<float>();
    auto velocity = new FrameBuffer<float>();

    auto configuration_map = new JSONConfigurationMap(R"(
                {
                    "wave": {
                        "physics": "acoustic",
                        "approximation": "isotropic",
                        "equation-order": "second",
                        "grid-sampling": "uniform"
                    },
                            "type": "none",
                            "properties": {
                                "boundary-saving": true
                             }
                }
            )"_json);
    auto memory_handler = new WaveFieldsMemoryHandler(configuration_map);

    float nt = 5;
    apGridBox->SetNT(nt);

    /*
     * Variables initialization for grid box.
     */

    int nx, ny, nz;
    int wnx, wnz, wny;

    nx = apGridBox->GetActualGridSize(X_AXIS);
    ny = apGridBox->GetActualGridSize(Y_AXIS);
    nz = apGridBox->GetActualGridSize(Z_AXIS);

    wnx = apGridBox->GetActualWindowSize(X_AXIS);
    wny = apGridBox->GetActualWindowSize(Y_AXIS);
    wnz = apGridBox->GetActualWindowSize(Z_AXIS);

    uint window_size = wnx * wny * wnz;
    uint size = nx * ny * nz;

    uint half_length = apParameters->GetHalfLength();
    uint bound_length = apParameters->GetBoundaryLength();
    uint offset = half_length + bound_length;

    uint start_y = 0;
    uint end_y = 1;

    uint start_z = offset;
    uint end_z = wnz - offset;
    uint start_x = offset;
    uint end_x = wnx - offset;

    if (ny > 1) {
        start_y = offset;
        end_y = wny - offset;
    }

    pressure_curr->Allocate(window_size);
    pressure_prev->Allocate(window_size);
    velocity->Allocate(size);

    apGridBox->RegisterWaveField(WAVE | GB_PRSS | CURR | DIR_Z, pressure_curr);
    apGridBox->RegisterWaveField(WAVE | GB_PRSS | PREV | DIR_Z, pressure_prev);
    apGridBox->RegisterWaveField(WAVE | GB_PRSS | NEXT | DIR_Z, pressure_prev);
    apGridBox->RegisterParameter(PARM | GB_VEL, velocity);

    float temp_vel[size];
    float dt = apGridBox->GetDT();

    for (int iy = 0; iy < ny; iy++) {
        for (int iz = 0; iz < nz; iz++) {
            for (int ix = 0; ix < nx; ix++) {
                temp_vel[iz * nx + ix + (iy * nx * nz)] = 1500;
                temp_vel[iz * nx + ix + (iy * nx * nz)] *=
                        temp_vel[iz * nx + ix + (iy * nx * nz)] * dt * dt;
            }
        }
    }
    Device::MemSet(pressure_curr->GetNativePointer(), 0.0f, window_size * sizeof(float));
    Device::MemSet(pressure_prev->GetNativePointer(), 0.0f, window_size * sizeof(float));
    Device::MemCpy(velocity->GetNativePointer(), temp_vel, size * sizeof(float), Device::COPY_HOST_TO_DEVICE);

    memory_handler->SetComputationParameters(apParameters);

    auto computation_kernel = new SecondOrderComputationKernel(configuration_map);
    computation_kernel->SetGridBox(apGridBox);
    computation_kernel->SetComputationParameters(apParameters);

    auto dependent_components_map = new ComponentsMap<DependentComponent>();
    dependent_components_map->Set(MEMORY_HANDLER, memory_handler);

    auto components_map = new ComponentsMap<Component>();
    components_map->Set(COMPUTATION_KERNEL, computation_kernel);


    auto forward_collector = new ReversePropagation(configuration_map);
    forward_collector->SetComputationParameters(apParameters);
    forward_collector->SetDependentComponents(dependent_components_map);
    forward_collector->SetComponentsMap(components_map);
    forward_collector->SetGridBox(apGridBox);
    forward_collector->AcquireConfiguration();

    /*
     * Test for rest grid case true , means the wva fields should be zero
     */

    Device::MemSet(pressure_curr->GetNativePointer(), 1.0f, window_size * sizeof(float));
    Device::MemSet(pressure_prev->GetNativePointer(), 1.0f, window_size * sizeof(float));

    forward_collector->ResetGrid(true);

    auto curr_pressure = apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    auto prev_pressure = apGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetHostPointer();

    int misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (curr_pressure[index] != 0) misses += 1;

            }
        }
    }
    REQUIRE(misses == 0);

    misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (prev_pressure[index] != 0) misses += 1;

            }
        }
    }
    REQUIRE(misses == 0);

    /*
     * Test for get forward grid  in false option
     * and case of reset grid false , it return the internal grd
     */

    forward_collector->ResetGrid(false);
    auto grid_box = forward_collector->GetForwardGrid();

    REQUIRE(grid_box->GetActualGridSize(X_AXIS) == nx);
    REQUIRE(grid_box->GetActualGridSize(Y_AXIS) == ny);
    REQUIRE(grid_box->GetActualGridSize(Z_AXIS) == nz);
    REQUIRE(grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(grid_box->Get(PARM | GB_VEL)->GetNativePointer() != nullptr);

    auto pres_curr = grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto pres_prev = grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();


    /*
     * Check for swapping pointers after applying reset grid with false option
     */

    forward_collector->ResetGrid(false);

    auto swap_grid_box = forward_collector->GetForwardGrid();

    auto swap_pres_curr = swap_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto swap_pres_prev = swap_grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();


    REQUIRE(swap_pres_curr == pres_prev);
    REQUIRE(swap_pres_prev == pres_curr);

    /*
    *testing fetch forward , means that a tep function is applied
    */

    auto fetch_grid_box = forward_collector->GetForwardGrid();

    auto h_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    int location = (wnx / 2) + (wnz / 2) * wnx + (wny / 2) * wnx * wnz;
    h_pressure[location] = 1;

    auto d_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    Device::MemCpy(d_pressure, h_pressure, window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);

    for (int it = 0; it < int(nt); it++) {
        forward_collector->SaveForward();
        forward_collector->FetchForward();
    }

    /*
     * Test save boundary and restore boundary
     */

    auto fetch_pres = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();

    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (fetch_pres[index] != 0) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses > 0);

    forward_collector->ResetGrid(false);
    auto backup_pressure = apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();

    uint index = 0;
    for (int iy = start_y; iy < end_y; iy++) {
        for (int iz = start_z; iz < end_z; iz++) {
            for (int ix = 0; ix < half_length; ix++) {
                backup_pressure[iy * wnz * wnx + iz * wnx + bound_length + ix] = 0.25;

                index++;
                backup_pressure[iy * wnz * wnx + iz * wnx + (wnx - bound_length - 1) - ix] = 0.25;

                index++;
            }
        }
    }
    for (int iy = start_y; iy < end_y; iy++) {
        for (int iz = 0; iz < half_length; iz++) {
            for (int ix = start_x; ix < end_x; ix++) {
                backup_pressure[iy * wnz * wnx + (bound_length + iz) * wnx + ix] = 0.25;

                index++;
                backup_pressure[iy * wnz * wnx + (wnz - bound_length - 1 - iz) * wnx + ix] = 0.25;

                index++;
            }
        }
    }
    if (ny > 1) {
        for (int iy = 0; iy < half_length; iy++) {
            for (int iz = start_z; iz < end_z; iz++) {
                for (int ix = start_x; ix < end_x; ix++) {
                    backup_pressure[(bound_length + iy) * wnz * wnx + iz * wnx + ix] = 0.25;
                    index++;

                    backup_pressure[(wny - bound_length - 1 - iy) * wnz * wnx + iz * wnx + ix] = 0.25;
                    index++;
                }
            }
        }
    }

    Device::MemCpy(apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer(), backup_pressure,
                   window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);

    forward_collector->SaveForward();
    forward_collector->FetchForward();
    auto fetch_backup_grid = forward_collector->GetForwardGrid();
    auto fetch_backup_pres = fetch_backup_grid->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();

    misses = 0;
    for (int iy = start_y; iy < end_y; iy++) {
        for (int iz = start_z; iz < end_z; iz++) {
            for (int ix = 0; ix < half_length; ix++) {
                if (fetch_backup_pres[iy * wnz * wnx + iz * wnx + bound_length + ix] != 0.25) {
                    misses += 1;
                }
                if (fetch_backup_pres[iy * wnz * wnx + iz * wnx + (wnx - bound_length - 1) - ix] != 0.25) {
                    misses += 1;
                }
            }
        }
    }
    for (int iy = start_y; iy < end_y; iy++) {
        for (int iz = 0; iz < half_length; iz++) {
            for (int ix = start_x; ix < end_x; ix++) {
                if (fetch_backup_pres[iy * wnz * wnx + (bound_length + iz) * wnx + ix] != 0.25) {
                    misses += 1;
                }
                if (fetch_backup_pres[iy * wnz * wnx + (wnz - bound_length - 1 - iz) * wnx + ix] != 0.25) {
                    misses += 1;
                }
            }
        }
    }
    if (ny > 1) {
        for (int iy = 0; iy < half_length; iy++) {
            for (int iz = start_z; iz < end_z; iz++) {
                for (int ix = start_x; ix < end_x; ix++) {
                    if (fetch_backup_pres[(bound_length + iy) * wnz * wnx + iz * wnx + ix] != 0.25) {
                        misses += 1;
                    }
                    if (fetch_backup_pres[(wny - bound_length - 1 - iy) * wnz * wnx + iz * wnx + ix] != 0.250) {
                        misses += 1;
                    }
                }
            }
        }
    }
    REQUIRE (misses == 0);

    delete apGridBox;
    delete apParameters;
    delete apConfigurationMap;

    delete components_map;
    delete dependent_components_map;
    delete computation_kernel;
    delete forward_collector;
}

TEST_CASE("Reverse Forward Collector - 2D - No Window", "[No Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_REVERSE_NO_INJECTION(
            generate_grid_box(OP_TU_2D, OP_TU_NO_WIND),
            generate_computation_parameters(OP_TU_NO_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("Reverse Forward Collector - 2D - Window", "[Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_REVERSE_NO_INJECTION(
            generate_grid_box(OP_TU_2D, OP_TU_INC_WIND),
            generate_computation_parameters(OP_TU_INC_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("Reverse Forward Collector Injection - 2D - No Window", "[No Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_REVERSE_INC_INJECTION(
            generate_grid_box(OP_TU_2D, OP_TU_NO_WIND),
            generate_computation_parameters(OP_TU_NO_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("Reverse Forward Collector Injection - 2D - Window", "[Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_REVERSE_INC_INJECTION(
            generate_grid_box(OP_TU_2D, OP_TU_INC_WIND),
            generate_computation_parameters(OP_TU_INC_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}