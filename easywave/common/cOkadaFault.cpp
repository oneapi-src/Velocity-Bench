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

#include <stdio.h>
#include <string.h>

#include "utilits.h"
#include "cOkadaFault.h"
#include <cmath>

static const double Rearth = 6384.e+3;

//=========================================================================
// Constructor
cOkadaFault::cOkadaFault()
{
    // obligatory user parameters
    lat = lon = depth = strike = dip = rake = RealMax;
    // optional user parameters
    mw = slip = length = width = 0.;
    adjust_depth               = 0;
    // default values
    refpos = FLT_POS_C;
    mu     = 3.5e+10;
    // derivative parameters
    zbot = sind = cosd = sins = coss = tand = coslat = wp = dslip = sslip = 0;
    // flags
    checked = 0;
}

//=========================================================================
// Destructor
cOkadaFault::~cOkadaFault()
{
}

//=========================================================================
// read fault parameters from input string. read only. consistency check in separate method
int cOkadaFault::read(char *faultparam)
{
    char *cp, buf[64];
    int   ierr;

    // origin location
    cp = strstr(faultparam, "-location");
    if (cp) {
        if (sscanf(cp, "%*s %lf %lf %lf", &lon, &lat, &depth) != 3)
            return Err.post("cOkadaFault::read: position");
        depth *= 1000;
    }

    // adjust depth if fault breaks through surface
    cp = strstr(faultparam, "-adjust_depth");
    if (cp)
        adjust_depth = 1;

    // reference point
    cp = strstr(faultparam, "-refpos");
    if (cp) {
        if (sscanf(cp, "%*s %s", buf) != 1)
            return Err.post("cOkadaFault::read: refpos");
        if (!strcmp(buf, "C") || !strcmp(buf, "c"))
            refpos = FLT_POS_C;
        else if (!strcmp(buf, "MT") || !strcmp(buf, "mt"))
            refpos = FLT_POS_MT;
        else if (!strcmp(buf, "BT") || !strcmp(buf, "bt"))
            refpos = FLT_POS_BT;
        else if (!strcmp(buf, "BB") || !strcmp(buf, "bb"))
            refpos = FLT_POS_BB;
        else if (!strcmp(buf, "MB") || !strcmp(buf, "mb"))
            refpos = FLT_POS_MB;
        else
            return Err.post("cOkadaFault::read: refpos");
    }

    // magnitude
    cp = strstr(faultparam, "-mw");
    if (cp)
        if (sscanf(cp, "%*s %lf", &mw) != 1)
            return Err.post("cOkadaFault::read: mw");

    // slip
    cp = strstr(faultparam, "-slip");
    if (cp) {
        if (sscanf(cp, "%*s %lf", &slip) != 1)
            return Err.post("cOkadaFault::read: slip");
        if (slip < 1.e-6)
            slip = 1.e-6;
    }

    // strike
    cp = strstr(faultparam, "-strike");
    if (cp)
        if (sscanf(cp, "%*s %lf", &strike) != 1)
            return Err.post("cOkadaFault::read: strike");

    // dip
    cp = strstr(faultparam, "-dip");
    if (cp)
        if (sscanf(cp, "%*s %lf", &dip) != 1)
            return Err.post("cOkadaFault::read: dip");

    // rake
    cp = strstr(faultparam, "-rake");
    if (cp)
        if (sscanf(cp, "%*s %lf", &rake) != 1)
            return Err.post("cOkadaFault::read: rake");

    // length and width
    cp = strstr(faultparam, "-size");
    if (cp) {
        if (sscanf(cp, "%*s %lf %lf", &length, &width) != 2)
            return Err.post("cOkadaFault::read: size");
        length *= 1000;
        width *= 1000;
    }

    // rigidity
    cp = strstr(faultparam, "-rigidity");
    if (cp)
        if (sscanf(cp, "%*s %lf", &mu) != 1)
            return Err.post("cOkadaFault::read: rigidity");

    // check fault data for integrity
    // ierr = check(); if(ierr) return ierr;

    return 0;
}

//================================================================================
int cOkadaFault::check()
// Check readed fault parameters for consistency and calculate secondary parameters
{

    // check necessary parameters
    if (lon == RealMax) {
        Err.post("cOkadaFault::check: lon");
        return FLT_ERR_DATA;
    }
    if (lat == RealMax) {
        Err.post("cOkadaFault::check: lat");
        return FLT_ERR_DATA;
    }
    if (depth == RealMax) {
        Err.post("cOkadaFault::check: depth");
        return FLT_ERR_DATA;
    }
    if (strike == RealMax) {
        Err.post("cOkadaFault::check: strike");
        return FLT_ERR_STRIKE;
    }
    if (rake == RealMax) {
        Err.post("cOkadaFault::check: rake");
        return FLT_ERR_DATA;
    }
    if (dip == RealMax) {
        Err.post("cOkadaFault::check: dip");
        return FLT_ERR_DATA;
    }

    // cache trigonometric expressions
    sind   = sindeg(dip);
    cosd   = cosdeg(dip);
    sins   = sindeg(90 - strike);
    coss   = cosdeg(90 - strike);
    tand   = tandeg(dip);
    coslat = cosdeg(lat);

    // branching through given parameters (the full solution table see end of file)
    if (!mw && !slip) {
        Err.post("cOkadaFault::check: not enough data");
        return FLT_ERR_DATA;
    } else if (!mw && slip) {
        if (!length && !width) {
            Err.post("cOkadaFault::check: not enough data");
            return FLT_ERR_DATA;
        } else if (length && !width) {
            width = length / 2;
        } else if (!length && width) {
            length = 2 * width;
        } else if (length && width) {
        } else {
            Err.post("cOkadaFault::check: internal error");
            return FLT_ERR_INTERNAL;
        }

        mw = 2. / 3. * (log10(mu * length * width * slip) - 9.1);
    } else if (mw && !slip) {
        if (!length && !width) {
            // scaling relations used by JMA
            length = pow(10., -1.80 + 0.5 * mw) * 1000;
            width  = length / 2;
        } else if (length && !width) {
            width = length / 2;
        } else if (!length && width) {
            length = 2 * width;
        } else if (length && width) {
        } else {
            Err.post("cOkadaFault::check: internal error");
            return FLT_ERR_INTERNAL;
        }
        slip = mw2m0() / mu / length / width;
    } else if (mw && slip) {
        if (!length && !width) {
            double area = mw2m0() / mu / slip;
            length      = sqrt(2 * area);
            width       = length / 2;
        } else if (length && !width) {
            width = mw2m0() / mu / slip / length;
        } else if (!length && width) {
            length = mw2m0() / mu / slip / width;
        } else if (length && width) {
            if (fabs(1 - mu * slip * length * width / mw2m0()) > 0.01) {
                Err.post("cOkadaFault::check: data inconsistency");
                return FLT_ERR_DATA;
            }
        } else {
            Err.post("cOkadaFault::check: internal error");
            return FLT_ERR_INTERNAL;
        }
    }

    // calculate bottom of the fault
    switch (refpos) {
        double ztop;
    case FLT_POS_C:
        ztop = depth - width / 2 * sind;
        if (ztop < 0) {
            if (adjust_depth) {
                ztop  = 0.;
                depth = ztop + width / 2 * sind;
            } else {
                Err.post("cOkadaFault::check: negative ztop");
                return FLT_ERR_ZTOP;
            }
        }
        zbot = depth + width / 2 * sind;
        break;
    case FLT_POS_MT:
    case FLT_POS_BT:
        zbot = depth + width * sind;
        break;
    case FLT_POS_BB:
    case FLT_POS_MB:
        ztop = depth - width * sind;
        if (ztop < 0) {
            if (adjust_depth) {
                ztop  = 0.;
                depth = ztop + width * sind;
            } else {
                Err.post("cOkadaFault::check: negative ztop");
                return FLT_ERR_ZTOP;
            }
        }
        zbot = depth;
        break;
    }

    // slip components
    dslip = slip * sindeg(rake);
    sslip = slip * cosdeg(rake);

    // surface projection of width
    wp = width * cosd;

    checked = 1;

    return 0;
}

//=========================================================================
double cOkadaFault::mw2m0()
{
    return pow(10., 3. * mw / 2 + 9.1);
}

//=========================================================================
double cOkadaFault::getM0()
{
    if (!checked) {
        Err.post("cOkadaFault::getM0: fault not checked");
        return -RealMax;
    }

    return mw2m0();
}

//=========================================================================
double cOkadaFault::getMw()
{
    if (!checked) {
        Err.post("cOkadaFault::getMw: fault not checked");
        return -RealMax;
    }

    return mw;
}

//=========================================================================
double cOkadaFault::getZtop()
{
    return (zbot - width * sind);
}

//=========================================================================
int cOkadaFault::global2local(double glon, double glat, double &lx, double &ly)
// from global geographical coordinates into local Okada's coordinates
{
    double x, y;

    // center coordinate system to refpos (lon/lat), convert to meters
    y = Rearth * g2r(glat - lat);
    x = Rearth * coslat * g2r(glon - lon);

    // rotate according to strike
    lx = x * coss + y * sins;
    ly = -x * sins + y * coss;

    // finally shift to Okada's origin point (BB)
    switch (refpos) {
    case FLT_POS_C:
        lx = lx + length / 2;
        ly = ly + wp / 2;
        break;
    case FLT_POS_MT:
        lx = lx + length / 2;
        ly = ly + wp;
        break;
    case FLT_POS_BT:
        ly = ly + wp;
        break;
    case FLT_POS_BB:
        break;
    case FLT_POS_MB:
        lx = lx + length / 2;
        break;
    }

    return 0;
}

//=========================================================================
int cOkadaFault::local2global(double lx, double ly, double &glon, double &glat)
// from local Okada's coordinates to global geographical
{
    double x(0.0), y(0.0), gx(0.0), gy(0.0);

    // define local coordinates relative to the fault refpos
    switch (refpos) {
    case FLT_POS_C:
        x = lx - length / 2;
        y = ly - wp / 2;
        break;
    case FLT_POS_MT:
        x = lx - length / 2;
        y = ly - wp;
        break;
    case FLT_POS_BT:
        x = lx;
        y = ly - wp;
        break;
    case FLT_POS_BB:
        x = lx;
        y = ly;
        break;
    case FLT_POS_MB:
        x = lx - length / 2;
        y = ly;
        break;
    }

    // back-rotate to geographical axes (negative strike!). Values are still in meters!
    gx = x * coss + y * (-sins);
    gy = -x * (-sins) + y * coss;

    // convert meters to degrees. This is offset in degrees relative to refpos. Add refpos coordinates for absolute values
    glat = r2g(gy / Rearth) + lat;
    glon = r2g(gx / Rearth / cosdeg(lat)) + lon;

    return 0;
}

//=========================================================================
// get ruptured area to calculate surface displacement
// parameter rand accounts for lateral enlargement at rands of rupture surface projection
int cOkadaFault::getDeformArea(double &lonmin, double &lonmax, double &latmin, double &latmax)
{
#define FLT_NL 2 // significant deformation area along length (in length units)
#define FLT_NW 5 // significant deformation area along width (in width units)
    int    ierr;
    double dxC, dyC, l2, w2, glon, glat;

    if (!checked) {
        Err.post("cOkadaFault::getDeformArea: attempt with non-checked fault");
        return FLT_ERR_INTERNAL;
    }

    // follow rectangle around the fault
    dxC = FLT_NL * length;
    l2  = length / 2;
    dyC = FLT_NW * wp;
    w2  = wp / 2;

    local2global(dxC + l2, dyC + w2, glon, glat);
    lonmin = lonmax = glon;
    latmin = latmax = glat;

    local2global(-dxC + l2, dyC + w2, glon, glat);
    if (glon < lonmin)
        lonmin = glon;
    if (glon > lonmax)
        lonmax = glon;
    if (glat < latmin)
        latmin = glat;
    if (glat > latmax)
        latmax = glat;

    local2global(-dxC + l2, -dyC + w2, glon, glat);
    if (glon < lonmin)
        lonmin = glon;
    if (glon > lonmax)
        lonmax = glon;
    if (glat < latmin)
        latmin = glat;
    if (glat > latmax)
        latmax = glat;

    local2global(dxC + l2, -dyC + w2, glon, glat);
    if (glon < lonmin)
        lonmin = glon;
    if (glon > lonmax)
        lonmax = glon;
    if (glat < latmin)
        latmin = glat;
    if (glat > latmax)
        latmax = glat;

    return 0;
}

//=========================================================================
int cOkadaFault::calculate(double lon0, double lat0, double &uz, double &ulon, double &ulat)
{
    int    ierr;
    double x, y, ux, uy;

    if (!checked) {
        Err.post("cOkadaFault::calculate: attempt with non-checked fault");
        return FLT_ERR_INTERNAL;
    }

    global2local(lon0, lat0, x, y);

    // Okada model
    okada(length, width, zbot, sind, cosd, sslip, dslip, x, y, 1, &ux, &uy, &uz);

    // back-rotate horizontal deformations to global coordinates (negative strike!)
    ulon = ux * coss + uy * (-sins);
    ulat = -ux * (-sins) + uy * coss;

    return 0;
}

//=========================================================================
int cOkadaFault::calculate(double lon0, double lat0, double &uz)
{
    int    ierr;
    double x, y, ux, uy;

    if (!checked) {
        Err.post("cOkadaFault::calculate: attempt with non-checked fault");
        return FLT_ERR_INTERNAL;
    }

    global2local(lon0, lat0, x, y);

    // Okada model
    okada(length, width, zbot, sind, cosd, sslip, dslip, x, y, 0, &ux, &uy, &uz);

    return 0;
}

//======================================================================================
// Input parameter selection: Solution table
// if given:
// mw slip L W  action
// -  -  -  -  err_data
// -  -  -  +  err_data
// -  -  +  -  err_data
// -  -  +  +  err_data
// -  +  -  -  err_data
// -  +  -  +  L=2W, Mw
// -  +  +  -  W=L/2, Mw
// -  +  +  +  Mw
// +  -  -  -  W&C96(L,W), S
// +  -  -  +  L=2W, S
// +  -  +  -  W=L/2, S
// +  -  +  +  S
// +  +  -  -  area(Mw), L=2W
// +  +  -  +  L
// +  +  +  -  W
// +  +  +  +  check Mw=muSLW
