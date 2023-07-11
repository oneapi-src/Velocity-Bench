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
// Created by ahmed-ayyad on 24/01/2021.
//

#include <operations/components/independents/concrete/trace-managers/SeismicTraceManager.hpp>

#include <operations/common/DataTypes.h>
#include <operations/utils/io/write_utils.h>
#include <operations/test-utils/dummy-data-generators/DummyConfigurationMapGenerator.hpp>
#include <operations/test-utils/dummy-data-generators/DummyGridBoxGenerator.hpp>
#include <operations/test-utils/dummy-data-generators/DummyParametersGenerator.hpp>
#include <operations/test-utils/dummy-data-generators/DummyTraceGenerator.hpp>
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
using namespace operations::utils::io;

#define TRACE_STRIDE_X 3
#define TRACE_STRIDE_Y 4
#define TRACE_STRIDE_Z 0


void TEST_CASE_TRACE_MANAGER(GridBox *apGridBox,
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


    auto uut = new SeismicTraceManager(apConfigurationMap);

    uut->SetComputationParameters(apParameters);
    uut->SetGridBox(apGridBox);
    uut->AcquireConfiguration();

    /*
     * Generates a dummy *.segy file
     */

    std::string file_name = OPERATIONS_TEST_DATA_PATH "/dummy_trace.segy";
    auto ground_truth = generate_dummy_trace(file_name, apGridBox,
                                             TRACE_STRIDE_X,
                                             TRACE_STRIDE_Y);

    std::vector<std::string> files;
    files.push_back(file_name);

    auto shots = uut->GetWorkingShots(files, 0, wnx * wny, "CSR");

    int theoretical_shot_count = (wnx / TRACE_STRIDE_X);

    if (wny > 1) {
        theoretical_shot_count *= (wny / TRACE_STRIDE_Y);
    }
    int shot_count = shots.size();
    REQUIRE(shot_count != 0);
    REQUIRE(shot_count == theoretical_shot_count);

    int shot_id = 0;

    uut->ReadShot(files, shots[shot_id], "CSR");
    uut->PreprocessShot(100);


    REQUIRE(uut->GetTracesHolder()->Traces != nullptr);
    REQUIRE(uut->GetTracesHolder()->PositionsX != nullptr);
    REQUIRE(uut->GetTracesHolder()->PositionsY != nullptr);

    auto source_point = uut->GetSourcePoint();
    int start_idx = apParameters->GetBoundaryLength() + apParameters->GetHalfLength();

    REQUIRE(source_point->x == start_idx);
    REQUIRE(source_point->z == start_idx);
    REQUIRE(source_point->y == start_idx * (apGridBox->GetLogicalGridSize(Y_AXIS) != 1));

    REQUIRE(uut->GetTracesHolder()->PositionsX[0] == start_idx);
    REQUIRE(uut->GetTracesHolder()->PositionsY[0] == start_idx * (apGridBox->GetLogicalGridSize(Y_AXIS) != 1));
    REQUIRE(uut->GetTracesHolder()->SampleNT == wnz);
    REQUIRE(uut->GetTracesHolder()->TraceSizePerTimeStep == 1);
    REQUIRE(uut->GetTracesHolder()->ReceiversCountX == 1);
    REQUIRE(uut->GetTracesHolder()->ReceiversCountY == 1);

    int stride_x = wnx / TRACE_STRIDE_X;
    for (int i = 0; i < wnz; i++) {
        auto diff = ground_truth[i * stride_x + shot_id] - uut->GetTracesHolder()->Traces[i];
        REQUIRE(diff == Approx(diff).margin(std::numeric_limits<float>::epsilon()));
    }

    uint pos = (apParameters->GetBoundaryLength() + apParameters->GetHalfLength()) *
               wnx + uut->GetTracesHolder()->PositionsX[0] +
               uut->GetTracesHolder()->PositionsY[0] * wnx * wnz;

    uut->ApplyTraces(apGridBox->GetNT() - 1);

    float test_value = uut->GetTracesHolder()->Traces[wnz - 1] * 1500 * 1500 * dt * dt;

    REQUIRE(approximately_equal(pressure_curr->GetHostPointer()[pos], test_value));

    uut->ApplyTraces(1);

    test_value += uut->GetTracesHolder()->Traces[0] * 1500 * 1500 * dt * dt;
    REQUIRE(approximately_equal(pressure_curr->GetHostPointer()[pos], test_value));

    uut->ApplyTraces(apGridBox->GetNT() / 2);

    test_value += uut->GetTracesHolder()->Traces[wnz / 2] * 1500 * 1500 * dt * dt;
    REQUIRE(approximately_equal(pressure_curr->GetHostPointer()[pos], test_value));

    remove(file_name.c_str());

    delete apGridBox;
    delete apParameters;
    delete apConfigurationMap;
}

TEST_CASE("SeismicTraceManager - 2D - No Window", "[No Window],[2D]") {
    TEST_CASE_TRACE_MANAGER(
            generate_grid_box(OP_TU_2D, OP_TU_NO_WIND),
            generate_computation_parameters(OP_TU_NO_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("SeismicTraceManager - 2D - Window", "[Window],[2D]") {
    TEST_CASE_TRACE_MANAGER(
            generate_grid_box(OP_TU_2D, OP_TU_INC_WIND),
            generate_computation_parameters(OP_TU_INC_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}
