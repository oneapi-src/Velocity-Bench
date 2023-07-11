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
// Created by ahmed-ayyad on 01/18/2020.
//

#include <operations/components/independents/concrete/migration-accommodators/CrossCorrelationKernel.hpp>
#include <operations/data-units/concrete/holders/FrameBuffer.hpp>

#include <operations/common/DataTypes.h>
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


void TEST_CASE_CROSS_CORRELATION_COMBINED_COMPENSATION(GridBox *apGridBox,
                                                       ComputationParameters *apParameters,
                                                       ConfigurationMap *apConfigurationMap) {
    apConfigurationMap->WriteValue(OP_K_PROPRIETIES, OP_K_COMPENSATION, OP_K_COMPENSATION_COMBINED);
    /*
     * Environment setting (i.e. Backend setting initialization).
     */
    set_environment();

    /*
     * Register and allocate parameters and wave fields in
     * grid box according to the current test case.
     */

    auto pressure_back = new FrameBuffer<float>();
    auto pressure_forward = new FrameBuffer<float>();

    auto *forward_gridbox = new GridBox;

    apGridBox->SetNT(1);
    apGridBox->Clone(forward_gridbox);

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

    pressure_back->Allocate(window_size);
    pressure_forward->Allocate(window_size);

    apGridBox->RegisterWaveField(WAVE | GB_PRSS | CURR | DIR_Z, pressure_back);
    forward_gridbox->RegisterWaveField(WAVE | GB_PRSS | CURR | DIR_Z, pressure_forward);

    Device::MemSet(pressure_back->GetNativePointer(), 0.0f, window_size * sizeof(float));
    Device::MemSet(pressure_forward->GetNativePointer(), 0.0f, window_size * sizeof(float));

    auto uut = new CrossCorrelationKernel(apConfigurationMap);

    uut->SetComputationParameters(apParameters);
    uut->SetGridBox(apGridBox);
    uut->AcquireConfiguration();

    float temp_a[window_size];
    float temp_b[window_size];
    float ground_truth[window_size];
    float ground_truth_1[window_size];
    float ground_truth_2[window_size];

    for (int i = 0; i < window_size; i++) {
        temp_a[i] = (float) rand() * 100 / RAND_MAX;
        temp_b[i] = (float) rand() * 100 / RAND_MAX;
        ground_truth[i] = temp_a[i] * temp_b[i];
        ground_truth_1[i] = temp_a[i] * temp_a[i];
        ground_truth_2[i] = temp_b[i] * temp_b[i];
    }

    Device::MemCpy(apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer(), temp_b,
                   window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);
    Device::MemCpy(forward_gridbox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer(), temp_a,
                   window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);

    uut->Correlate(forward_gridbox);

    auto correlation_result = uut->GetShotCorrelation()->GetHostPointer();
    int misses = 0;

    uint offset = apParameters->GetHalfLength();
    uint nxEnd = wnx - offset;
    uint nyEnd = 1;
    uint nzEnd = wnz - offset;
    int y_start = 0;
    if (ny != 1) {
        y_start = offset;
        nyEnd = wny - offset;
    }
    /// Loop over correlation test case
    for (int j = y_start; j < nyEnd; j++) {
        for (int k = offset; k < nzEnd; k++) {
            for (int i = offset; i < nxEnd; i++) {
                int index = j * wnx * wnz + k * wnx + i;
                misses += !approximately_equal(correlation_result[index], ground_truth[index]);
            }
        }
    }

    REQUIRE(misses == 0);

    misses = 0;

    uut->Stack();

    auto stack_result = uut->GetStackedShotCorrelation()->GetHostPointer();

    offset = apParameters->GetHalfLength() + apParameters->GetBoundaryLength();
    nxEnd = wnx - offset;
    nzEnd = wnz - offset;
    if (ny != 1) {
        y_start = offset;
        nyEnd = wny - offset;
    }

    /// Loop over stacked correlation test case
    for (int i = offset; i < nxEnd; i++) {
        for (int j = y_start; j < nyEnd; j++) {
            for (int k = offset; k < nzEnd; k++) {
                int window_index = j * wnx * wnz + k * wnx + i;
                int grid_index = j * nx * nz + k * nx + i;
                misses += !approximately_equal(stack_result[grid_index], ground_truth[window_index]);
            }
        }
    }
    REQUIRE(misses == 0);

    misses = 0;

    uut->ResetShotCorrelation();
    correlation_result = uut->GetShotCorrelation()->GetHostPointer();

    for (int i = 0; i < window_size; i++) {
        misses += correlation_result[i] != 0.0;
    }
    REQUIRE(misses == 0);
    misses = 0;

    for (int i = 0; i < window_size; i++) {
        temp_a[i] = (float) rand() * 100 / RAND_MAX;
        temp_b[i] = (float) rand() * 100 / RAND_MAX;
        ground_truth[i] += temp_a[i] * temp_b[i];
        ground_truth_1[i] += temp_a[i] * temp_a[i];
        ground_truth_2[i] += temp_b[i] * temp_b[i];
    }
    Device::MemCpy(forward_gridbox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer(), temp_a,
                   window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);
    Device::MemCpy(apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer(), temp_b,
                   window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);

    uut->Correlate(forward_gridbox);
    uut->Stack();
    stack_result = uut->GetStackedShotCorrelation()->GetHostPointer();

    /// Loop over stacked correlation test case - 2nd round
    for (int i = offset; i < nxEnd; i++) {
        for (int j = y_start; j < nyEnd; j++) {
            for (int k = offset; k < nzEnd; k++) {
                int window_index = j * wnx * wnz + k * wnx + i;
                int grid_index = j * nx * nz + k * nx + i;
                misses += !approximately_equal(stack_result[grid_index], ground_truth[window_index]);
            }
        }
    }

    REQUIRE(misses == 0);
    misses = 0;

    auto migration_result = uut->GetMigrationData();

    REQUIRE(migration_result->GetGridSize(X_AXIS) == apGridBox->GetLogicalGridSize(X_AXIS));
    REQUIRE(migration_result->GetGridSize(Y_AXIS) == apGridBox->GetLogicalGridSize(Y_AXIS));
    REQUIRE(migration_result->GetGridSize(Z_AXIS) == apGridBox->GetLogicalGridSize(Z_AXIS));
    REQUIRE(migration_result->GetNT() == apGridBox->GetNT());

    REQUIRE(migration_result->GetCellDimensions(X_AXIS) == apGridBox->GetCellDimensions(X_AXIS));
    REQUIRE(migration_result->GetCellDimensions(Y_AXIS) == apGridBox->GetCellDimensions(Y_AXIS));
    REQUIRE(migration_result->GetCellDimensions(Z_AXIS) == apGridBox->GetCellDimensions(Z_AXIS));
    REQUIRE(migration_result->GetDT() == apGridBox->GetDT());
    REQUIRE(migration_result->GetGatherDimension() == 1);

    REQUIRE(migration_result->GetResults().size() == 3);

    auto migration_buffer = migration_result->GetResultAt(0)->GetData();
    auto source_illumination_buffer = migration_result->GetResultAt(1)->GetData();
    auto receiver_illumination_buffer = migration_result->GetResultAt(2)->GetData();

    /// Loop over migration buffer test case - 2nd round
    for (int i = offset; i < nxEnd; i++) {
        for (int j = y_start; j < nyEnd; j++) {
            for (int k = offset; k < nzEnd; k++) {
                int window_index = j * wnx * wnz + k * wnx + i;
                int grid_index = j * nx * nz + k * nx + i;

                misses += !approximately_equal(migration_buffer[grid_index], ground_truth[window_index]);
                misses += !approximately_equal(source_illumination_buffer[grid_index], ground_truth_1[window_index]);
                misses += !approximately_equal(receiver_illumination_buffer[grid_index], ground_truth_2[window_index]);
            }
        }
    }
    REQUIRE(misses == 0);

    delete apGridBox;
    delete apParameters;
    delete apConfigurationMap;

    delete pressure_back;
    delete pressure_forward;
    delete forward_gridbox;

    delete uut;
}


void TEST_CASE_CROSS_CORRELATION_NO_COMPENSATION(GridBox *apGridBox,
                                                 ComputationParameters *apParameters,
                                                 ConfigurationMap *apConfigurationMap) {
    apConfigurationMap->WriteValue(OP_K_PROPRIETIES, OP_K_COMPENSATION, OP_K_COMPENSATION_NONE);

    /*
     * Environment setting (i.e. Backend setting initialization).
     */
    set_environment();

    /*
     * Register and allocate parameters and wave fields in
     * grid box according to the current test case.
     */

    auto pressure_back = new FrameBuffer<float>();
    auto pressure_forward = new FrameBuffer<float>();

    auto forward_gridbox = new GridBox;

    apGridBox->SetNT(1);
    apGridBox->Clone(forward_gridbox);

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

    pressure_back->Allocate(window_size);
    pressure_forward->Allocate(window_size);

    apGridBox->RegisterWaveField(WAVE | GB_PRSS | CURR | DIR_Z, pressure_back);
    forward_gridbox->RegisterWaveField(WAVE | GB_PRSS | CURR | DIR_Z, pressure_forward);

    Device::MemSet(pressure_back->GetNativePointer(), 0.0f, window_size * sizeof(float));
    Device::MemSet(pressure_forward->GetNativePointer(), 0.0f, window_size * sizeof(float));

    auto uut = new CrossCorrelationKernel(apConfigurationMap);

    uut->SetComputationParameters(apParameters);
    uut->SetGridBox(apGridBox);
    uut->AcquireConfiguration();

    float temp_a[window_size];
    float temp_b[window_size];
    float ground_truth[window_size];

    for (int i = 0; i < window_size; i++) {
        temp_a[i] = (float) rand() * 100 / RAND_MAX;
        temp_b[i] = (float) rand() * 100 / RAND_MAX;
        ground_truth[i] = temp_a[i] * temp_b[i];
    }

    Device::MemCpy(apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer(), temp_b,
                   window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);
    Device::MemCpy(forward_gridbox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer(), temp_a,
                   window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);

    uut->Correlate(forward_gridbox);

    auto correlation_result = uut->GetShotCorrelation()->GetHostPointer();
    int misses = 0;

    uint offset = apParameters->GetHalfLength();
    uint nxEnd = wnx - offset;
    uint nyEnd = 1;
    uint nzEnd = wnz - offset;
    int y_start = 0;
    if (ny != 1) {
        y_start = offset;
        nyEnd = wny - offset;
    }

    /// Loop over correlation test case
    for (int j = y_start; j < nyEnd; j++) {
        for (int k = offset; k < nzEnd; k++) {
            for (int i = offset; i < nxEnd; i++) {
                int index = j * wnx * wnz + k * wnx + i;
                misses += !approximately_equal(correlation_result[index], ground_truth[index]);
            }
        }
    }
    REQUIRE(misses == 0);

    misses = 0;

    uut->Stack();

    auto stack_result = uut->GetStackedShotCorrelation()->GetHostPointer();

    offset = apParameters->GetHalfLength() + apParameters->GetBoundaryLength();
    nxEnd = wnx - offset;
    nzEnd = wnz - offset;
    if (ny != 1) {
        y_start = offset;
        nyEnd = wny - offset;
    }

    /// Loop over stacked correlation test case
    for (int i = offset; i < nxEnd; i++) {
        for (int j = y_start; j < nyEnd; j++) {
            for (int k = offset; k < nzEnd; k++) {
                int window_index = j * wnx * wnz + k * wnx + i;
                int grid_index = j * nx * nz + k * nx + i;
                misses += !approximately_equal(stack_result[grid_index], ground_truth[window_index]);
            }
        }
    }
    REQUIRE(misses == 0);

    misses = 0;

    uut->ResetShotCorrelation();
    correlation_result = uut->GetShotCorrelation()->GetHostPointer();

    for (int i = 0; i < window_size; i++) {
        misses += correlation_result[i] != 0.0;
    }
    REQUIRE(misses == 0);
    misses = 0;

    for (int i = 0; i < window_size; i++) {
        temp_a[i] = (float) rand() * 100 / RAND_MAX;
        temp_b[i] = (float) rand() * 100 / RAND_MAX;
        ground_truth[i] += temp_a[i] * temp_b[i];
    }
    Device::MemCpy(forward_gridbox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer(), temp_a,
                   window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);
    Device::MemCpy(apGridBox->Get(WAVE | GB_PRSS | CURR | DIR_Z)->GetNativePointer(), temp_b,
                   window_size * sizeof(float), Device::COPY_HOST_TO_DEVICE);

    uut->Correlate(forward_gridbox);
    uut->Stack();
    stack_result = uut->GetStackedShotCorrelation()->GetHostPointer();

    /// Loop over stacked correlation test case - 2nd round
    for (int i = offset; i < nxEnd; i++) {
        for (int j = y_start; j < nyEnd; j++) {
            for (int k = offset; k < nzEnd; k++) {
                int window_index = j * wnx * wnz + k * wnx + i;
                int grid_index = j * nx * nz + k * nx + i;
                misses += !approximately_equal(stack_result[grid_index], ground_truth[window_index]);
            }
        }
    }
    REQUIRE(misses == 0);
    misses = 0;

    auto migration_result = uut->GetMigrationData();

    REQUIRE(migration_result->GetGridSize(X_AXIS) == apGridBox->GetLogicalGridSize(X_AXIS));
    REQUIRE(migration_result->GetGridSize(Y_AXIS) == apGridBox->GetLogicalGridSize(Y_AXIS));
    REQUIRE(migration_result->GetGridSize(Z_AXIS) == apGridBox->GetLogicalGridSize(Z_AXIS));
    REQUIRE(migration_result->GetNT() == apGridBox->GetNT());

    REQUIRE(migration_result->GetCellDimensions(X_AXIS) == apGridBox->GetCellDimensions(X_AXIS));
    REQUIRE(migration_result->GetCellDimensions(Y_AXIS) == apGridBox->GetCellDimensions(Y_AXIS));
    REQUIRE(migration_result->GetCellDimensions(Z_AXIS) == apGridBox->GetCellDimensions(Z_AXIS));
    REQUIRE(migration_result->GetDT() == apGridBox->GetDT());
    REQUIRE(migration_result->GetGatherDimension() == 1);

    REQUIRE(migration_result->GetResults().size() == 1);

    auto migration_buffer = migration_result->GetResultAt(0)->GetData();

    /// Loop over migration buffer test case - 2nd round
    for (int i = offset; i < nxEnd; i++) {
        for (int j = y_start; j < nyEnd; j++) {
            for (int k = offset; k < nzEnd; k++) {
                int window_index = j * wnx * wnz + k * wnx + i;
                int grid_index = j * nx * nz + k * nx + i;
                misses += !approximately_equal(migration_buffer[grid_index], ground_truth[window_index]);
            }
        }
    }
    REQUIRE(misses == 0);

    delete apGridBox;
    delete apParameters;
    delete apConfigurationMap;

    delete pressure_back;
    delete pressure_forward;
    delete forward_gridbox;

    delete uut;
}

TEST_CASE("CrossCorrelation - No Compensation - 2D - No Window", "[No Window],[2D]") {
    TEST_CASE_CROSS_CORRELATION_NO_COMPENSATION(
            generate_grid_box(OP_TU_2D, OP_TU_NO_WIND),
            generate_computation_parameters(OP_TU_NO_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("CrossCorrelation - No Compensation - 2D - Window", "[Window],[2D]") {
    TEST_CASE_CROSS_CORRELATION_NO_COMPENSATION(
            generate_grid_box(OP_TU_2D, OP_TU_INC_WIND),
            generate_computation_parameters(OP_TU_INC_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("CrossCorrelation - Combined Compensation - 2D - No Window", "[No Window],[2D]") {
    TEST_CASE_CROSS_CORRELATION_COMBINED_COMPENSATION(
            generate_grid_box(OP_TU_2D, OP_TU_NO_WIND),
            generate_computation_parameters(OP_TU_NO_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("CrossCorrelation - Combined Compensation - 2D - Window", "[Window],[2D]") {
    TEST_CASE_CROSS_CORRELATION_COMBINED_COMPENSATION(
            generate_grid_box(OP_TU_2D, OP_TU_INC_WIND),
            generate_computation_parameters(OP_TU_INC_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}