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

#include <operations/components/independents/concrete/forward-collectors/TwoPropagation.hpp>

#include <operations/common/DataTypes.h>
#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>
#include <operations/test-utils/dummy-data-generators/DummyConfigurationMapGenerator.hpp>
#include <operations/test-utils/dummy-data-generators/DummyGridBoxGenerator.hpp>
#include <operations/test-utils/dummy-data-generators/DummyParametersGenerator.hpp>
#include <operations/test-utils/NumberHelpers.hpp>
#include <operations/test-utils/EnvironmentHandler.hpp>

#include <libraries/catch/catch.hpp>

using namespace std;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;
using namespace operations::configuration;
using namespace operations::testutils;
using namespace operations::helpers;


void TEST_CASE_FORWARD_COLLECTOR_TWO(GridBox *apGridBox,
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

    auto memory_handler = new WaveFieldsMemoryHandler(apConfigurationMap);

    float nt = 300;
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

    memory_handler->SetComputationParameters(apParameters);
    auto dependent_components_map = new ComponentsMap<DependentComponent>();
    dependent_components_map->Set(MEMORY_HANDLER, memory_handler);

    auto configuration_map = new JSONConfigurationMap(R"(

                {
                   "properties": {
                        "zfp-tolerance": 0.9,
                        "zfp-parallel": 4,
                        "zfp-relative": false,
                        "write-path": "./test-data",
                        "compression-type": "zfp",
                        "compression": false
                    }

                }
            )"_json);
    auto forward_collector = new TwoPropagation(configuration_map);
    forward_collector->SetComputationParameters(apParameters);
    forward_collector->SetDependentComponents(dependent_components_map);
    forward_collector->SetGridBox(apGridBox);
    forward_collector->AcquireConfiguration();

    /*
     * Test for rest grid case true , means the wva fields should be zero
     */

    Device::MemSet(pressure_curr->GetNativePointer(), 1.0f, window_size * sizeof(float));
    Device::MemSet(pressure_prev->GetNativePointer(), 1.0f, window_size * sizeof(float));

    int misses = 0;

    forward_collector->ResetGrid(true);
    auto grid_box = forward_collector->GetForwardGrid();

    auto curr_pressure = apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    auto prev_pressure = apGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetHostPointer();

    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (curr_pressure[index] != 0) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses == 0);

    /*
     * Check that the wave fields are set to zero
     */

    misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (prev_pressure[index] != 0) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses == 0);

    forward_collector->ResetGrid(false);
    auto init_grid_box = forward_collector->GetForwardGrid();

    REQUIRE(init_grid_box->GetActualGridSize(X_AXIS) == nx);
    REQUIRE(init_grid_box->GetActualGridSize(Y_AXIS) == ny);
    REQUIRE(init_grid_box->GetActualGridSize(Z_AXIS) == nz);
    REQUIRE(init_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(init_grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(init_grid_box->Get(PARM | GB_VEL)->GetNativePointer() != nullptr);

    auto pres_curr = grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto pres_prev = grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();
    auto pres_next = grid_box->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();

    forward_collector->ResetGrid(false);

    /*
     * Check that the pointers are swapped
     */

    auto swap_grid_box = forward_collector->GetForwardGrid();

    auto swap_pres_curr = swap_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto swap_pres_prev = swap_grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();
    auto swap_pres_next = swap_grid_box->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();

    REQUIRE(swap_pres_prev == pres_next);

    auto fetch_grid_box = forward_collector->GetForwardGrid();

    auto h_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    int location = (wnx / 2) + (wnz / 2) * wnx + (wny / 2) * wnx * wnz;
    h_pressure[location] = 1;

    auto d_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    Device::MemCpy(d_pressure, h_pressure, window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);


    for (int it = 0; it < int(nt); it++) {
        forward_collector->SaveForward();
    }

    auto fetch_pres = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();

    /*
     * Check that there is a propagation
     */

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

    for (int it = int(nt); it > 0; it--) {
        forward_collector->FetchForward();
    }

    auto fetch_backup_grid_box = forward_collector->GetForwardGrid();
    auto fetch_pres_backup = fetch_backup_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();

    /*
     * Check that the arrays is stored in a file , and the data from the file read is the same output of saving
     */

    misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (fetch_pres[index] != fetch_pres_backup[index]) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses > 0);

    delete apGridBox;
    delete apParameters;
    delete apConfigurationMap;

    delete dependent_components_map;
    delete memory_handler;
    delete forward_collector;
}

void TEST_CASE_FORWARD_COLLECTOR_TWO_INC_COMPRESSION_NO_TOLERANCE(GridBox *apGridBox,
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

    float nt = 300;
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

    auto memory_handler = new WaveFieldsMemoryHandler(apConfigurationMap);
    memory_handler->SetComputationParameters(apParameters);
    auto dependent_components_map = new ComponentsMap<DependentComponent>();
    dependent_components_map->Set(MEMORY_HANDLER, memory_handler);

    auto configuration_map = new JSONConfigurationMap(R"(
                {
                   "properties": {
                        "zfp-tolerance": 0.0,
                        "zfp-parallel": 1,
                        "zfp-relative": true,
                        "write-path": "test",
                        "compression-type": "zfp",
                        "compression": true
                    }

                }
            )"_json);
    auto forward_collector = new TwoPropagation(configuration_map);
    forward_collector->SetComputationParameters(apParameters);
    forward_collector->SetDependentComponents(dependent_components_map);
    forward_collector->SetGridBox(apGridBox);
    forward_collector->AcquireConfiguration();

    /*
     * Test for rest grid case true , means the wva fields should be zero
     */

    Device::MemSet(pressure_curr->GetNativePointer(), 1.0f, window_size * sizeof(float));
    Device::MemSet(pressure_prev->GetNativePointer(), 1.0f, window_size * sizeof(float));

    forward_collector->ResetGrid(true);
    auto grid_box = forward_collector->GetForwardGrid();

    auto curr_pressure = apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    auto prev_pressure = apGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetHostPointer();

    int misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (curr_pressure[index] != 0) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses == 0);

    misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (prev_pressure[index] != 0) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses == 0);

    forward_collector->ResetGrid(false);
    auto init_grid_box = forward_collector->GetForwardGrid();

    REQUIRE(init_grid_box->GetActualGridSize(X_AXIS) == nx);
    REQUIRE(init_grid_box->GetActualGridSize(Y_AXIS) == ny);
    REQUIRE(init_grid_box->GetActualGridSize(Z_AXIS) == nz);
    REQUIRE(init_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(init_grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(init_grid_box->Get(PARM | GB_VEL)->GetNativePointer() != nullptr);

    auto pres_curr = grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto pres_prev = grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();
    auto pres_next = grid_box->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();

    forward_collector->ResetGrid(false);
    auto swap_grid_box = forward_collector->GetForwardGrid();

    auto swap_pres_curr = swap_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto swap_pres_prev = swap_grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();
    auto swap_pres_next = swap_grid_box->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();

    REQUIRE(swap_pres_prev == pres_next);

    auto fetch_grid_box = forward_collector->GetForwardGrid();

    auto h_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    int location = (wnx / 2) + (wnz / 2) * wnx + (wny / 2) * wnx * wnz;
    h_pressure[location] = 1;

    auto d_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    Device::MemCpy(d_pressure, h_pressure, window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);

    for (int it = 0; it < int(nt); it++) {
        forward_collector->SaveForward();
    }

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

    for (int it = int(nt); it > 0; it--) {
        forward_collector->FetchForward();
    }

    auto fetch_backup_grid_box = forward_collector->GetForwardGrid();
    auto fetch_pres_backup = fetch_backup_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();

    misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (fetch_pres[index] != fetch_pres_backup[index]) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses > 0);

    delete apGridBox;
    delete apParameters;
    delete apConfigurationMap;

    delete dependent_components_map;
    delete memory_handler;
    delete forward_collector;
}

void TEST_CASE_FORWARD_COLLECTOR_TWO_INC_COMPRESSION_NO_RELATIVE(GridBox *apGridBox,
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
                   "properties": {
                        "zfp-tolerance": 0.9,
                        "zfp-relative": false,
                        "write-path": "test",
                        "zfp-parallel": 2,
                        "compression-type": "zfp",
                        "compression": true
                    }

                }
            )"_json);

    float nt = 300;
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

    auto memory_handler = new WaveFieldsMemoryHandler(apConfigurationMap);
    memory_handler->SetComputationParameters(apParameters);
    auto dependent_components_map = new ComponentsMap<DependentComponent>();
    dependent_components_map->Set(MEMORY_HANDLER, memory_handler);

    auto forward_collector = new TwoPropagation(configuration_map);
    forward_collector->SetComputationParameters(apParameters);
    forward_collector->SetDependentComponents(dependent_components_map);
    forward_collector->SetGridBox(apGridBox);
    forward_collector->AcquireConfiguration();

    /*
     * Test for rest grid case true , means the wva fields should be zero
     */

    Device::MemSet(pressure_curr->GetNativePointer(), 1.0f, window_size * sizeof(float));
    Device::MemSet(pressure_prev->GetNativePointer(), 1.0f, window_size * sizeof(float));

    forward_collector->ResetGrid(true);
    auto grid_box = forward_collector->GetForwardGrid();

    auto curr_pressure = apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    auto prev_pressure = apGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetHostPointer();

    int misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (curr_pressure[index] != 0) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses == 0);

    misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (prev_pressure[index] != 0) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses == 0);

    forward_collector->ResetGrid(false);
    auto init_grid_box = forward_collector->GetForwardGrid();

    REQUIRE(init_grid_box->GetActualGridSize(X_AXIS) == nx);
    REQUIRE(init_grid_box->GetActualGridSize(Y_AXIS) == ny);
    REQUIRE(init_grid_box->GetActualGridSize(Z_AXIS) == nz);
    REQUIRE(init_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(init_grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(init_grid_box->Get(PARM | GB_VEL)->GetNativePointer() != nullptr);

    auto pres_curr = grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto pres_prev = grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();
    auto pres_next = grid_box->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();


    forward_collector->ResetGrid(false);

    auto swap_grid_box = forward_collector->GetForwardGrid();

    auto swap_pres_curr = swap_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto swap_pres_prev = swap_grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();
    auto swap_pres_next = swap_grid_box->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();

    REQUIRE(swap_pres_prev == pres_next);

    auto fetch_grid_box = forward_collector->GetForwardGrid();

    auto h_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    int location = (wnx / 2) + (wnz / 2) * wnx + (wny / 2) * wnx * wnz;
    h_pressure[location] = 1;

    auto d_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    Device::MemCpy(d_pressure, h_pressure, window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);

    for (int it = 0; it < int(nt); it++) {
        forward_collector->SaveForward();
    }

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

    for (int it = int(nt); it > 0; it--) {
        forward_collector->FetchForward();
    }

    auto fetch_backup_grid_box = forward_collector->GetForwardGrid();
    auto fetch_pres_backup = fetch_backup_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();

    misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (fetch_pres[index] != fetch_pres_backup[index]) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses > 0);

    delete apGridBox;
    delete apParameters;
    delete apConfigurationMap;

    delete dependent_components_map;
    delete memory_handler;
    delete forward_collector;
}

void TEST_CASE_FORWARD_COLLECTOR_TWO_INC_COMPRESSION_NO_PARALLEL(GridBox *apGridBox,
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

    float nt = 300;
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

    auto memory_handler = new WaveFieldsMemoryHandler(apConfigurationMap);
    memory_handler->SetComputationParameters(apParameters);
    auto dependent_components_map = new ComponentsMap<DependentComponent>();
    dependent_components_map->Set(MEMORY_HANDLER, memory_handler);

    auto configuration_map = new JSONConfigurationMap(R"(

                {
                   "properties": {
                        "zfp-tolerance": 0.9,
                        "zfp-relative": true,
                        "write-path": "test",
                        "zfp-parallel": 1,
                        "compression-type": "zfp",
                        "compression": true
                    }

                }
            )"_json);
    auto forward_collector = new TwoPropagation(configuration_map);
    forward_collector->SetComputationParameters(apParameters);
    forward_collector->SetDependentComponents(dependent_components_map);
    forward_collector->SetGridBox(apGridBox);
    forward_collector->AcquireConfiguration();

    /*
     * Test for rest grid case true , means the wva fields should be zero
     */

    Device::MemSet(pressure_curr->GetNativePointer(), 1.0f, window_size * sizeof(float));
    Device::MemSet(pressure_prev->GetNativePointer(), 1.0f, window_size * sizeof(float));

    forward_collector->ResetGrid(true);
    auto grid_box = forward_collector->GetForwardGrid();

    auto curr_pressure = apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    auto prev_pressure = apGridBox->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetHostPointer();

    int misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (curr_pressure[index] != 0) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses == 0);

    misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (prev_pressure[index] != 0) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses == 0);

    forward_collector->ResetGrid(false);
    auto init_grid_box = forward_collector->GetForwardGrid();

    REQUIRE(init_grid_box->GetActualGridSize(X_AXIS) == nx);
    REQUIRE(init_grid_box->GetActualGridSize(Y_AXIS) == ny);
    REQUIRE(init_grid_box->GetActualGridSize(Z_AXIS) == nz);
    REQUIRE(init_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(init_grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer() != nullptr);
    REQUIRE(init_grid_box->Get(PARM | GB_VEL)->GetNativePointer() != nullptr);

    auto pres_curr = grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto pres_prev = grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();
    auto pres_next = grid_box->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();

    forward_collector->ResetGrid(false);

    auto swap_grid_box = forward_collector->GetForwardGrid();

    auto swap_pres_curr = swap_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    auto swap_pres_prev = swap_grid_box->Get(WAVE | GB_PRSS | PREV | DIR_Z)->GetNativePointer();
    auto swap_pres_next = swap_grid_box->Get(WAVE | GB_PRSS | NEXT | DIR_Z)->GetNativePointer();

    REQUIRE(swap_pres_prev == pres_next);

    auto fetch_grid_box = forward_collector->GetForwardGrid();

    auto h_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();
    int location = (wnx / 2) + (wnz / 2) * wnx + (wny / 2) * wnx * wnz;
    h_pressure[location] = 1;

    auto d_pressure = fetch_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer();
    Device::MemCpy(d_pressure, h_pressure, window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);


    for (int it = 0; it < int(nt); it++) {
        forward_collector->SaveForward();
    }

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

    for (int it = int(nt); it > 0; it--) {
        forward_collector->FetchForward();
    }

    auto fetch_backup_grid_box = forward_collector->GetForwardGrid();
    auto fetch_pres_backup = fetch_backup_grid_box->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetHostPointer();

    misses = 0;
    for (int iy = 0; iy < wny; iy++) {
        for (int iz = 0; iz < wnz; iz++) {
            for (int ix = 0; ix < wnx; ix++) {
                int index = iy * wnx * wnz + iz * wnx + ix;
                if (fetch_pres[index] != fetch_pres_backup[index]) {
                    misses += 1;
                }
            }
        }
    }
    REQUIRE(misses > 0);

    delete apGridBox;
    delete apParameters;
    delete apConfigurationMap;

    delete dependent_components_map;
    delete memory_handler;
    delete forward_collector;
}

TEST_CASE("Two Forward Collector - 2D - No Window", "[No Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_TWO(
            generate_grid_box(OP_TU_2D, OP_TU_NO_WIND),
            generate_computation_parameters(OP_TU_NO_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("Two Forward Collector - 2D - Window", "[Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_TWO(
            generate_grid_box(OP_TU_2D, OP_TU_INC_WIND),
            generate_computation_parameters(OP_TU_INC_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("Two Forward Collector Injection no tolerance - 2D - No Window", "[No Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_TWO_INC_COMPRESSION_NO_TOLERANCE(
            generate_grid_box(OP_TU_2D, OP_TU_NO_WIND),
            generate_computation_parameters(OP_TU_NO_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("TwoForward Collector Injection no tolerance  - 2D - Window", "[Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_TWO_INC_COMPRESSION_NO_TOLERANCE(
            generate_grid_box(OP_TU_2D, OP_TU_INC_WIND),
            generate_computation_parameters(OP_TU_INC_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("Two Forward Collector Injection no relative - 2D - No Window", "[No Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_TWO_INC_COMPRESSION_NO_RELATIVE(
            generate_grid_box(OP_TU_2D, OP_TU_NO_WIND),
            generate_computation_parameters(OP_TU_NO_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("TwoForward Collector Injection no relative  - 2D - Window", "[Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_TWO_INC_COMPRESSION_NO_RELATIVE(
            generate_grid_box(OP_TU_2D, OP_TU_INC_WIND),
            generate_computation_parameters(OP_TU_INC_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("Two Forward Collector Injection no parallel - 2D - No Window", "[No Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_TWO_INC_COMPRESSION_NO_PARALLEL(
            generate_grid_box(OP_TU_2D, OP_TU_NO_WIND),
            generate_computation_parameters(OP_TU_NO_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("TwoForward Collector Injection no parallel  - 2D - Window", "[Window],[2D]") {
    TEST_CASE_FORWARD_COLLECTOR_TWO_INC_COMPRESSION_NO_PARALLEL(
            generate_grid_box(OP_TU_2D, OP_TU_INC_WIND),
            generate_computation_parameters(OP_TU_INC_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}
