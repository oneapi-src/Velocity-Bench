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
// Created by marwan-elsafty on 03/02/2021.
//

#include <operations/components/independents/concrete/model-handlers/SeismicModelHandler.hpp>

#include <operations/common/DataTypes.h>
#include <operations/test-utils/dummy-data-generators/DummyConfigurationMapGenerator.hpp>
#include <operations/test-utils/dummy-data-generators/DummyGridBoxGenerator.hpp>
#include <operations/test-utils/dummy-data-generators/DummyParametersGenerator.hpp>
#include <operations/test-utils/dummy-data-generators/DummyModelGenerator.hpp>
#include <operations/test-utils/NumberHelpers.hpp>
#include <operations/test-utils/EnvironmentHandler.hpp>

#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>
#include <operations/components/dependency/concrete/HasDependents.hpp>

#include <libraries/catch/catch.hpp>

#include <map>
#include <string>

using namespace std;
using namespace operations;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;
using namespace operations::configuration;
using namespace operations::testutils;
using namespace operations::helpers;


void TEST_CASE_SEISMIC_MODEL_HANDLER_CORE(GridBox *apGridBox,
                                          ComputationParameters *apParameters) {
    SECTION("LogicalGridSize") {
        REQUIRE(apGridBox->GetLogicalGridSize(X_AXIS) == 23);
        REQUIRE(apGridBox->GetLogicalGridSize(Y_AXIS) == 23);
        REQUIRE(apGridBox->GetLogicalGridSize(Z_AXIS) == 23);
    }
    SECTION("CellDimensions") {
        REQUIRE(apGridBox->GetCellDimensions(X_AXIS) == Approx(10));
        REQUIRE(apGridBox->GetCellDimensions(Y_AXIS) == Approx(10));
        REQUIRE(apGridBox->GetCellDimensions(Z_AXIS) == Approx(10));
    }
    SECTION("ReferencePoint") {
        REQUIRE(apGridBox->GetReferencePoint(X_AXIS) == 0);
        REQUIRE(apGridBox->GetReferencePoint(Y_AXIS) == 0);
        REQUIRE(apGridBox->GetReferencePoint(Z_AXIS) == 0);
    }
    SECTION("WindowStart") {
        REQUIRE(apGridBox->GetWindowStart(X_AXIS) == 0);
        REQUIRE(apGridBox->GetWindowStart(Y_AXIS) == 0);
        REQUIRE(apGridBox->GetWindowStart(Z_AXIS) == 0);
    }
    SECTION("LogicalWindowSize") {
        REQUIRE(apGridBox->GetLogicalWindowSize(X_AXIS) == 23);
        REQUIRE(apGridBox->GetLogicalWindowSize(Y_AXIS) == 23);
        REQUIRE(apGridBox->GetLogicalWindowSize(Z_AXIS) == 23);
    }
    SECTION("ActualGridSize") {
#if defined(USING_DPCPP)
        if (apParameters->IsUsingWindow()) {
            REQUIRE(apGridBox->GetActualGridSize(X_AXIS) == 23);
        } else {
            REQUIRE(apGridBox->GetActualGridSize(X_AXIS) == 32);
        }
#else
        REQUIRE(apGridBox->GetActualGridSize(X_AXIS) == 23);
#endif
        REQUIRE(apGridBox->GetActualGridSize(Y_AXIS) == 23);
        REQUIRE(apGridBox->GetActualGridSize(Z_AXIS) == 23);
    }

    SECTION("ActualWindowSize") {
#if defined(USING_DPCPP)
        REQUIRE(apGridBox->GetActualWindowSize(X_AXIS) == 32);
#else
        REQUIRE(apGridBox->GetActualWindowSize(X_AXIS) == 23);
#endif
        REQUIRE(apGridBox->GetActualWindowSize(Y_AXIS) == 23);
        REQUIRE(apGridBox->GetActualWindowSize(Z_AXIS) == 23);
    }
    SECTION("ComputationGridSize") {
        REQUIRE(apGridBox->GetComputationGridSize(X_AXIS) == 15);
        REQUIRE(apGridBox->GetComputationGridSize(Y_AXIS) == 15);
        REQUIRE(apGridBox->GetComputationGridSize(Z_AXIS) == 15);
    }
}

void TEST_CASE_SEISMIC_MODEL_HANDLER_APPROXIMATION(GridBox *apGridBox,
                                                   ComputationParameters *apParameters) {
    SECTION("Wave Allocation") {
        REQUIRE(apGridBox->Get(WAVE | GB_PRSS | CURR)->GetNativePointer() != nullptr);
        REQUIRE(apGridBox->Get(WAVE | GB_PRSS | PREV)->GetNativePointer() != nullptr);
        REQUIRE(apGridBox->Get(WAVE | GB_PRSS | NEXT)->GetNativePointer() != nullptr);
    }

    if (apParameters->GetApproximation() == ISOTROPIC) {
        SECTION("Parameters Allocation") {
            REQUIRE(apGridBox->Get(PARM | GB_VEL)->GetNativePointer() != nullptr);
            REQUIRE(apGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer() != nullptr);
        }
    } else if (apParameters->GetApproximation() == VTI) {
        SECTION("Parameters Allocation") {
            REQUIRE(apGridBox->Get(PARM | GB_VEL)->GetNativePointer() != nullptr);
            REQUIRE(apGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer() != nullptr);

            REQUIRE(apGridBox->Get(PARM | GB_EPS)->GetNativePointer() != nullptr);
            REQUIRE(apGridBox->Get(PARM | WIND | GB_EPS)->GetNativePointer() != nullptr);

            REQUIRE(apGridBox->Get(PARM | GB_DLT)->GetNativePointer() != nullptr);
            REQUIRE(apGridBox->Get(PARM | WIND | GB_DLT)->GetNativePointer() != nullptr);
        }
    } else if (apParameters->GetApproximation() == TTI) {
        SECTION("Parameters Allocation") {
            REQUIRE(apGridBox->Get(PARM | GB_VEL)->GetNativePointer() != nullptr);
            REQUIRE(apGridBox->Get(PARM | WIND | GB_VEL)->GetNativePointer() != nullptr);

            REQUIRE(apGridBox->Get(PARM | GB_EPS)->GetNativePointer() != nullptr);
            REQUIRE(apGridBox->Get(PARM | WIND | GB_EPS)->GetNativePointer() != nullptr);

            REQUIRE(apGridBox->Get(PARM | GB_EPS)->GetNativePointer() != nullptr);
            REQUIRE(apGridBox->Get(PARM | WIND | GB_EPS)->GetNativePointer() != nullptr);

            REQUIRE(apGridBox->Get(PARM | GB_THT)->GetNativePointer() != nullptr);
            REQUIRE(apGridBox->Get(PARM | WIND | GB_THT)->GetNativePointer() != nullptr);

            REQUIRE(apGridBox->Get(PARM | GB_PHI)->GetNativePointer() != nullptr);
            REQUIRE(apGridBox->Get(PARM | WIND | GB_PHI)->GetNativePointer() != nullptr);
        }
    }
}

void TEST_CASE_SEISMIC_MODEL_HANDLER(GridBox *apGridBox,
                                     ComputationParameters *apParameters,
                                     ConfigurationMap *apConfigurationMap) {
    /*
     * Environment setting (i.e. Backend setting initialization).
     */
    set_environment();

    auto model_handler = new SeismicModelHandler(apConfigurationMap);
    model_handler->SetComputationParameters(apParameters);

    auto memory_handler = new WaveFieldsMemoryHandler(apConfigurationMap);
    memory_handler->SetComputationParameters(apParameters);

    auto dependant_components_map = new helpers::ComponentsMap<components::DependentComponent>();
    dependant_components_map->Set(MEMORY_HANDLER, memory_handler);
    model_handler->SetDependentComponents(dependant_components_map);

    /*
     * Generates a dummy *.segy file
     */

//    generate_dummy_model("dummy_model");
//
//    map<std::string, std::string> file_names;
//    file_names["velocity"] = OPERATIONS_TEST_DATA_PATH "/dummy_model.segy";
//
//    auto grid_box = model_handler->ReadModel(file_names);
//
//    remove(OPERATIONS_TEST_DATA_PATH "/dummy_model.segy");

    printf("Deprecated. Model Handler is not tested\n");
//    TEST_CASE_SEISMIC_MODEL_HANDLER_CORE(grid_box, apParameters);
//    TEST_CASE_SEISMIC_MODEL_HANDLER_APPROXIMATION(grid_box, apParameters);

//    delete grid_box;
    delete model_handler;
    delete memory_handler;
    delete dependant_components_map;

    delete apGridBox;
    delete apParameters;
    delete apConfigurationMap;
}

/*
 * Isotropic Test Cases
 */

TEST_CASE("SeismicModelHandler - 2D - No Window - ISO", "[No Window],[2D],[ISO]") {
    TEST_CASE_SEISMIC_MODEL_HANDLER(
            generate_grid_box(OP_TU_2D, OP_TU_NO_WIND),
            generate_computation_parameters(OP_TU_NO_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}

TEST_CASE("SeismicModelHandler - 2D - Window - ISO", "[Window],[2D],[ISO]") {
    TEST_CASE_SEISMIC_MODEL_HANDLER(
            generate_grid_box(OP_TU_2D, OP_TU_INC_WIND),
            generate_computation_parameters(OP_TU_INC_WIND, ISOTROPIC),
            generate_average_case_configuration_map_wave());
}
