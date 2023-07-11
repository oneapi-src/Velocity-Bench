/*
 * EasyWave - A realtime tsunami simulation program with GPU support.
 * Copyright (C) 2014  Andrey Babeyko, Johannes Spazier
 * GFZ German Research Centre for Geosciences (http://www.gfz-potsdam.de)
 *
 * Parts of this program (especially the GPU extension) were developed
 * within the context of the following publicly funded project:
 * - TRIDEC, EU 7th Framework Programme, Grant Agreement 258723
 *   (http://www.tridec-online.eu)
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent versions of the EUPL (the "Licence"),
 * complemented with the following provision: For the scientific transparency
 * and verification of results obtained and communicated to the public after
 * using a modified version of the work, You (as the recipient of the source
 * code and author of this modified version, used to produce the published
 * results in scientific communications) commit to make this modified source
 * code available in a repository that is easily and freely accessible for a
 * duration of five years after the communication of the obtained results.
 *
 * You may not use this work except in compliance with the Licence.
 *
 * You may obtain a copy of the Licence at:
 * https://joinup.ec.europa.eu/software/page/eupl
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 */

/*
 * Modifications Copyright (C) 2023 Intel Corporation
 * 
 * This Program is subject to the terms of the European Union Public License 1.2
 * 
 * If a copy of the license was not distributed with this file, you can obtain one at 
 * https://joinup.ec.europa.eu/sites/default/files/custom-page/attachment/2020-03/EUPL-1.2%20EN.txt
 * 
 * SPDX-License-Identifier: EUPL-1.2
 */

// Y. Okada (1985) Surface deformation due to shear and tensile faults in a half-space:
// Bull.Seism.Soc.Am., v.75, p.1135-1154.
// okada@bosai.go.jp

#include <math.h>
#define My_PI    3.14159265358979
#define DISPLMAX 1000

double fun_Chinnery(double (*fun)(double ksi, double eta), double x, double y);
double f_ssUx(double ksi, double eta);
double f_ssUy(double ksi, double eta);
double f_ssUz(double ksi, double eta);
double f_dsUx(double ksi, double eta);
double f_dsUy(double ksi, double eta);
double f_dsUz(double ksi, double eta);
double fun_R(double ksi, double eta);
double fun_X(double ksi, double eta);
double fun_yp(double ksi, double eta);
double fun_dp(double ksi, double eta);
double fun_I1(double ksi, double eta);
double fun_I2(double ksi, double eta);
double fun_I3(double ksi, double eta);
double fun_I4(double ksi, double eta);
double fun_I5(double ksi, double eta);

static double sdip;
static double cdip;
static double p;
static double q;
static double width;
static double length;
static double elast;

//============================================================================
int okada(double L, double W, double D, double sinD, double cosD, double U1, double U2, double x, double y, int flag_xy, double *Ux, double *Uy, double *Uz)
{
    double U1x, U2x, U1y, U2y, U1z, U2z;

    sdip = sinD;
    if (fabs(sdip) < 1.e-10)
        sdip = 0;
    cdip = cosD;
    if (fabs(cdip) < 1.e-10)
        cdip = 0;
    p      = y * cdip + D * sdip;
    q      = y * sdip - D * cdip;
    width  = W;
    length = L;
    elast  = 0.5; // mu/(lambda+mu)

    U1x = U2x = U1y = U2y = U1z = U2z = 0;

    if (U1 != 0) {
        if (flag_xy) {
            U1x = -U1 / 2 / My_PI * fun_Chinnery(f_ssUx, x, y);
            if (fabs(U1x) > DISPLMAX)
                U1x = 0;
            U1y = -U1 / 2 / My_PI * fun_Chinnery(f_ssUy, x, y);
            if (fabs(U1y) > DISPLMAX)
                U1y = 0;
        }
        U1z = -U1 / 2 / My_PI * fun_Chinnery(f_ssUz, x, y);
        if (fabs(U1z) > DISPLMAX)
            U1z = 0;
    }

    if (U2 != 0) {
        if (flag_xy) {
            U2x = -U2 / 2 / My_PI * fun_Chinnery(f_dsUx, x, y);
            if (fabs(U2x) > DISPLMAX)
                U2x = 0;
            U2y = -U2 / 2 / My_PI * fun_Chinnery(f_dsUy, x, y);
            if (fabs(U2y) > DISPLMAX)
                U2y = 0;
        }
        U2z = -U2 / 2 / My_PI * fun_Chinnery(f_dsUz, x, y);
        if (fabs(U2z) > DISPLMAX)
            U2z = 0;
    }

    *Ux = U1x + U2x;
    *Uy = U1y + U2y;
    *Uz = U1z + U2z;

    return 0;
}

double fun_Chinnery(double (*fun)(double ksi, double eta), double x, double /*y*/)
{
    double value;

    value = fun(x, p) - fun(x, p - width) - fun(x - length, p) + fun(x - length, p - width);

    return value;
}

double f_ssUx(double ksi, double eta)
{
    double val, R, I1, term2;

    R  = fun_R(ksi, eta);
    I1 = fun_I1(ksi, eta);

    if (q * R == 0) {
        if (ksi * eta == 0) {
            term2 = 0;
        } else {
            if (ksi * eta * q * R > 0)
                term2 = My_PI;
            else
                term2 = -My_PI;
        }
    } else {
        term2 = atan(ksi * eta / q / R);
    }

    val = ksi * q / R / (R + eta) + term2 + I1 * sdip;

    return val;
}

double f_ssUy(double ksi, double eta)
{
    double val, yp, R, I2;

    R  = fun_R(ksi, eta);
    I2 = fun_I2(ksi, eta);
    yp = fun_yp(ksi, eta);

    val = yp * q / R / (R + eta) + q * cdip / (R + eta) + I2 * sdip;

    return val;
}

double f_ssUz(double ksi, double eta)
{
    double val, dp, R, I4;

    R  = fun_R(ksi, eta);
    I4 = fun_I4(ksi, eta);
    dp = fun_dp(ksi, eta);

    val = dp * q / R / (R + eta) + q * sdip / (R + eta) + I4 * sdip;

    return val;
}

double f_dsUx(double ksi, double eta)
{
    double val, R, I3;

    R  = fun_R(ksi, eta);
    I3 = fun_I3(ksi, eta);

    val = q / R - I3 * sdip * cdip;

    return val;
}

double f_dsUy(double ksi, double eta)
{
    double val, yp, R, I1, term2;

    R  = fun_R(ksi, eta);
    I1 = fun_I1(ksi, eta);
    yp = fun_yp(ksi, eta);

    if (q * R == 0) {
        if (ksi * eta == 0) {
            term2 = 0;
        } else {
            if (ksi * eta * q * R > 0)
                term2 = My_PI;
            else
                term2 = -My_PI;
        }
    } else {
        term2 = atan(ksi * eta / q / R);
    }

    val = yp * q / R / (R + ksi) + cdip * term2 - I1 * sdip * cdip;

    return val;
}

double f_dsUz(double ksi, double eta)
{
    double val, dp, R, I5, term2;

    R  = fun_R(ksi, eta);
    I5 = fun_I5(ksi, eta);
    dp = fun_dp(ksi, eta);

    if (q * R == 0) {
        if (ksi * eta == 0) {
            term2 = 0;
        } else {
            if (ksi * eta * q * R > 0)
                term2 = My_PI;
            else
                term2 = -My_PI;
        }
    } else {
        term2 = atan(ksi * eta / q / R);
    }

    val = dp * q / R / (R + ksi) + sdip * term2 - I5 * sdip * cdip;

    return val;
}

double fun_R(double ksi, double eta)
{
    double val;

    val = sqrt(ksi * ksi + eta * eta + q * q);

    return val;
}

double fun_X(double ksi, double /* eta */)
{
    double val;

    val = sqrt(ksi * ksi + q * q);

    return val;
}

double fun_dp(double /* ksi */, double eta)
{
    double val;

    val = eta * sdip - q * cdip;

    return val;
}

double fun_yp(double /* ksi */, double eta)
{
    double val;

    val = eta * cdip + q * sdip;

    return val;
}

double fun_I1(double ksi, double eta)
{
    double val, R, dp, I5;

    R  = fun_R(ksi, eta);
    dp = fun_dp(ksi, eta);
    I5 = fun_I5(ksi, eta);

    if (cdip != 0)
        val = elast * (-1. / cdip * ksi / (R + dp)) - sdip / cdip * I5;
    else
        val = -elast / 2 * ksi * q / (R + dp) / (R + dp);

    return val;
}

double fun_I2(double ksi, double eta)
{
    double val, R, I3;

    R  = fun_R(ksi, eta);
    I3 = fun_I3(ksi, eta);

    val = elast * (-log(R + eta)) - I3;

    return val;
}

double fun_I3(double ksi, double eta)
{
    double val, R, yp, dp, I4;

    R  = fun_R(ksi, eta);
    yp = fun_yp(ksi, eta);
    dp = fun_dp(ksi, eta);
    I4 = fun_I4(ksi, eta);

    if (cdip != 0)
        val = elast * (1. / cdip * yp / (R + dp) - log(R + eta)) + sdip / cdip * I4;
    else
        val = elast / 2 * (eta / (R + dp) + yp * q / (R + dp) / (R + dp) - log(R + eta));

    return val;
}

double fun_I4(double ksi, double eta)
{
    double val, R, dp;

    R  = fun_R(ksi, eta);
    dp = fun_dp(ksi, eta);

    if (cdip != 0)
        val = elast / cdip * (log(R + dp) - sdip * log(R + eta));
    else
        val = -elast * q / (R + dp);

    return val;
}

double fun_I5(double ksi, double eta)
{
    double val, dp, X, R;

    if (ksi == 0)
        return (double)0;

    R  = fun_R(ksi, eta);
    X  = fun_X(ksi, eta);
    dp = fun_dp(ksi, eta);

    if (cdip != 0)
        val = elast * 2 / cdip * atan((eta * (X + q * cdip) + X * (R + X) * sdip) / (ksi * (R + X) * cdip));
    else
        val = -elast * ksi * sdip / (R + dp);

    return val;
}
