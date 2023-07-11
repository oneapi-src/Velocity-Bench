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
// Created by zeyad-osama on 14/02/2021.
//

#include <operations/common/ComputationParameters.hpp>

#include <operations/test-utils/EnvironmentHandler.hpp>

#include <libraries/catch/catch.hpp>

using namespace std;
using namespace operations::common;
using namespace operations::testutils;

#define MARGIN std::numeric_limits<float>::epsilon()

void TEST_CASE_COMPUTATION_PARAMETERS() {
    /*
     * Environment setting (i.e. Backend setting initialization).
     */
    set_environment();

    auto cp = new ComputationParameters(O_2);

    auto hl = O_2;
    REQUIRE(hl == cp->GetHalfLength());

    auto order = SECOND;
    cp->SetEquationOrder(order);
    REQUIRE(order == cp->GetEquationOrder());

    auto physics = ACOUSTIC;
    cp->SetPhysics(physics);
    REQUIRE(physics == cp->GetPhysics());

    auto approximation = ISOTROPIC;
    cp->SetApproximation(approximation);
    REQUIRE(approximation == cp->GetApproximation());

    auto source_frequency = 12.5f;
    cp->SetSourceFrequency(source_frequency);
    REQUIRE(source_frequency == cp->GetSourceFrequency());

    auto isotropic_radius = 5;
    cp->SetIsotropicRadius(isotropic_radius);
    REQUIRE(isotropic_radius == cp->GetIsotropicRadius());

    auto boundary_length = 5;
    cp->SetBoundaryLength(boundary_length);
    REQUIRE(boundary_length == cp->GetBoundaryLength());

    auto relaxed_dt = 0.9f;
    cp->SetRelaxedDT(relaxed_dt);
    REQUIRE(relaxed_dt == cp->GetRelaxedDT());

    auto using_wind = true;
    cp->SetIsUsingWindow(using_wind);
    REQUIRE(using_wind == cp->IsUsingWindow());

    auto wind_left = 1;
    cp->SetLeftWindow(wind_left);
    REQUIRE(wind_left == cp->GetLeftWindow());

    auto wind_right = 2;
    cp->SetRightWindow(wind_right);
    REQUIRE(wind_right == cp->GetRightWindow());

    auto wind_depth = 3;
    cp->SetDepthWindow(wind_depth);
    REQUIRE(wind_depth == cp->GetDepthWindow());

    auto wind_back = 4;
    cp->SetBackWindow(wind_back);
    REQUIRE(wind_back == cp->GetBackWindow());

    auto wind_front = 5;
    cp->SetFrontWindow(wind_front);
    REQUIRE(wind_front == cp->GetFrontWindow());

    auto algorithm = RTM;
    cp->SetAlgorithm(algorithm);
    REQUIRE(algorithm == cp->GetAlgorithm());

    auto block_x = 1;
    cp->SetBlockX(block_x);
    REQUIRE(block_x == cp->GetBlockX());

    auto block_y = 2;
    cp->SetBlockY(block_y);
    REQUIRE(block_y == cp->GetBlockY());

    auto block_z = 3;
    cp->SetBlockZ(block_z);
    REQUIRE(block_z == cp->GetBlockZ());

    auto thread_cnt = 4;
    cp->SetThreadCount(thread_cnt);
    REQUIRE(thread_cnt == cp->GetThreadCount());

    delete cp;
}

void TEST_CASE_COMPUTATION_PARAMETERS_FD_COEFFICIENTS(HALF_LENGTH aHalfLength) {
    /*
     * Environment setting (i.e. Backend setting initialization).
     */
    set_environment();

    auto cp = new ComputationParameters(aHalfLength);

    if (aHalfLength == O_2) {
        /*
         * Accuracy = 2
         * Half Length = 1
         */

        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[0] == -2);
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[1] == 1);

        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[0] == 0);
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[1] == 0.5);

        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[0] == 0.0f);
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[1] == 1.0f);
    } else if (aHalfLength == O_4) {
        /*
         * Accuracy = 4
         * Half Length = 2
         */

        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[0] ==
                Approx(-2.5).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[1] ==
                Approx(1.33333333333).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[2] ==
                Approx(-0.08333333333).margin(MARGIN));

        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[0] == 0.0f);
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[1] ==
                Approx(2.0f / 3.0f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[2] ==
                Approx(-1.0f / 12.0f).margin(MARGIN));

        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[0] == 0.0f);
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[1] ==
                Approx(1.125f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[2] ==
                Approx(-0.041666666666666664f).margin(MARGIN));
    } else if (aHalfLength == O_8) {
        /*
         * Accuracy = 8
         * Half Length = 4
         */

        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[0] ==
                Approx(-2.847222222).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[1] ==
                Approx(1.6).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[2] ==
                Approx(-0.2).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[3] ==
                Approx(+2.53968e-2).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[4] ==
                Approx(-1.785714e-3).margin(MARGIN));

        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[0] == 0);
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[1] ==
                Approx(0.8).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[2] ==
                Approx(-0.2).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[3] ==
                Approx(0.03809523809).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[4] ==
                Approx(-0.00357142857).margin(MARGIN));

        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[0] == 0.0f);
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[1] ==
                Approx(1.1962890625f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[2] ==
                Approx(-0.07975260416666667f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[3] ==
                Approx(0.0095703125f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[4] ==
                Approx(-0.0006975446428571429f).margin(MARGIN));
    } else if (aHalfLength == O_12) {
        /*
         * Accuracy = 12
         * Half Length = 6
         */

        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[0] ==
                Approx(-2.98277777778).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[1] ==
                Approx(1.71428571429).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[2] ==
                Approx(-0.26785714285).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[3] ==
                Approx(0.05291005291).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[4] ==
                Approx(-0.00892857142).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[5] ==
                Approx(0.00103896103).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[6] ==
                Approx(-0.00006012506).margin(MARGIN));

        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[0] == 0.0);
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[1] ==
                Approx(0.857142857143).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[2] ==
                Approx(-0.267857142857).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[3] ==
                Approx(0.0793650793651).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[4] ==
                Approx(-0.0178571428571).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[5] ==
                Approx(0.0025974025974).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[6] ==
                Approx(-0.000180375180375).margin(MARGIN));

        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[0] == 0.0f);
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[1] ==
                Approx(1.2213363647460938f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[2] ==
                Approx(-0.09693145751953125f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[3] ==
                Approx(0.017447662353515626f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[4] ==
                Approx(-0.002967289515904018f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[5] ==
                Approx(0.0003590053982204861f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[6] ==
                Approx(-2.184781161221591e-05f).margin(MARGIN));
    } else if (aHalfLength == O_16) {
        /*
         * Accuracy = 16
         * Half Length = 8
         */
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[0] ==
                Approx(-3.05484410431).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[1] ==
                Approx(1.77777777778).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[2] ==
                Approx(-0.311111111111).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[3] ==
                Approx(0.0754208754209).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[4] ==
                Approx(-0.0176767676768).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[5] ==
                Approx(0.00348096348096).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[6] ==
                Approx(-0.000518000518001).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[7] ==
                Approx(5.07429078858e-05).margin(MARGIN));
        REQUIRE(cp->GetSecondDerivativeFDCoefficient()[8] ==
                Approx(-2.42812742813e-06).margin(MARGIN));

        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[0] ==
                Approx(-6.93889390391e-17).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[1] ==
                Approx(0.888888888889).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[2] ==
                Approx(-0.311111111111).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[3] ==
                Approx(0.113131313131).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[4] ==
                Approx(-0.0353535353535).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[5] ==
                Approx(0.00870240870241).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[6] ==
                Approx(-0.001554001554).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[7] ==
                Approx(0.0001776001776).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeFDCoefficient()[8] ==
                Approx(-9.71250971251e-06).margin(MARGIN));

        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[0] == 0.0f);
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[1] ==
                Approx(1.2340910732746122f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[2] ==
                Approx(-0.10664984583854668f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[3] ==
                Approx(0.023036366701126076f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[4] ==
                Approx(-0.005342385598591385f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[5] ==
                Approx(0.0010772711700863268f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[6] ==
                Approx(-0.00016641887751492495f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[7] ==
                Approx(1.7021711056048922e-05f).margin(MARGIN));
        REQUIRE(cp->GetFirstDerivativeStaggeredFDCoefficient()[8] ==
                Approx(-8.523464202880773e-07f).margin(MARGIN));
    }
    delete cp;
}

TEST_CASE("Computation Parameters - Class",
          "[ComputationParameters],[Class]") {
    TEST_CASE_COMPUTATION_PARAMETERS();
}

TEST_CASE("Computation Parameters - FD Coefficients",
          "[ComputationParameters],[FDCoefficients]") {
    for (auto hl : {O_2, O_4, O_8, O_12, O_16}) {
        TEST_CASE_COMPUTATION_PARAMETERS_FD_COEFFICIENTS(hl);
    }
}