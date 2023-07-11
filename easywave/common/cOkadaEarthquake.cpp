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

// cOkadaEarthquake.cpp: implementation of the cOkadaEarthquake class
//
//=========================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utilits.h"
#include "cOkadaEarthquake.h"
#include <cmath>

//=========================================================================
cOkadaEarthquake::cOkadaEarthquake()
{
    finalized = 0;
    m0        = 0;
    nfault    = 1;
    fault     = new cOkadaFault[1];
}

//=========================================================================
cOkadaEarthquake::~cOkadaEarthquake()
{
    if (fault != nullptr)
        delete[] fault;
}

//=========================================================================
// Read fault(s) from a file
int cOkadaEarthquake::read(char *fname)
{
    FILE *fp;
    char  record[256];
    int   ierr;

    nfault = utlGetNumberOfRecords(fname);
    if (fault != NULL)
        delete[] fault;
    fault = new cOkadaFault[nfault];

    if ((fp = fopen(fname, "rt")) == NULL)
        return Err.post("Unable to open %s", fname);

    for (int n = 0, line = 0; n < nfault; n++) {
        ierr = utlReadNextRecord(fp, record, &line);
        if (ierr == EOF) {
            if (fp != nullptr)
                fclose(fp);
            return Err.post("%s:%d has an unexpected EOF", fname, line);
        }
        ierr = fault[n].read(record);
        if (ierr) {
            if (fp != nullptr)
                fclose(fp);
            return ierr;
        }
    }

    fclose(fp);

    return 0;
}

//=========================================================================
// check all ruptures for parameter integrity
int cOkadaEarthquake::finalizeInput()
{
    char msg[32];
    int  ierr, n;

    for (m0 = 0., n = 0; n < nfault; n++) {

        ierr = fault[n].check();

        if (ierr) {
            Err.post("Fault number: %d", n + 1);
            return (10 * n + ierr);
        }

        m0 += fault[n].getM0();
    }

    finalized = 1;

    return 0;
}

//=========================================================================
double cOkadaEarthquake::getM0()
{
    if (!finalized) {
        Err.post("cOkadaEarthquake::getM0: eq not finalized");
        return -RealMax;
    }

    return m0;
}

//=========================================================================
double cOkadaEarthquake::getMw()
{
    if (!finalized) {
        Err.post("cOkadaEarthquake::getMw: eq not finalized");
        return -RealMax;
    }

    return (2. / 3. * (log10(m0) - 9.1));
}

//=========================================================================
// get ruptured area to calculate surface displacement
int cOkadaEarthquake::getDeformArea(int round, double &lonmin, double &lonmax, double &latmin, double &latmax)
{
    int    ierr;
    double xmin, xmax, ymin, ymax;

    if (!finalized)
        return Err.post("cOkadaEarthquake::calculate: eq not finalized");

    lonmin = latmin = RealMax;
    lonmax = latmax = -RealMax;

    for (int n = 0; n < nfault; n++) {

        ierr = fault[n].getDeformArea(xmin, xmax, ymin, ymax);
        if (ierr)
            return ierr;

        if (xmin < lonmin)
            lonmin = xmin;
        if (xmax > lonmax)
            lonmax = xmax;
        if (ymin < latmin)
            latmin = ymin;
        if (ymax > latmax)
            latmax = ymax;
    }

    if (round) {
        lonmin = floor(lonmin);
        lonmax = ceil(lonmax);
        latmin = floor(latmin);
        latmax = ceil(latmax);
    }

    return 0;
}

//=========================================================================
// Calculate surface displacements by summing effect from all ruptures
int cOkadaEarthquake::calculate(double lon, double lat, double &uz, double &ulon, double &ulat)
{
    int    n, ierr;
    double uzf, ulonf, ulatf;

    if (!finalized)
        return Err.post("cOkadaEarthquake::calculate: eq not finalized");

    for (ulon = ulat = uz = 0, n = 0; n < nfault; n++) {
        ierr = fault[n].calculate(lon, lat, uzf, ulonf, ulatf);
        if (ierr)
            return ierr;
        uz += uzf;
        ulon += ulonf;
        ulat += ulatf;
    }

    return 0;
}

//=========================================================================
// Calculate surface displacements by summing effect from all ruptures
int cOkadaEarthquake::calculate(double lon, double lat, double &uz)
{
    int    n, ierr;
    double uzf;

    if (!finalized)
        return Err.post("cOkadaEarthquake::calculate: eq not finalized");

    for (uz = 0, n = 0; n < nfault; n++) {
        ierr = fault[n].calculate(lon, lat, uzf);
        if (ierr)
            return ierr;
        uz += uzf;
    }

    return 0;
}

//=========================================================================
// Calculate surface displacements by summing effect from all ruptures
int cOkadaEarthquake::calculate(cOgrd &uZ)
{
    int    i, j, n, ierr;
    double uzf;

    if (!finalized)
        return Err.post("cOkadaEarthquake::calculate: eq not finalized");
    ierr = setGrid(uZ);
    if (ierr)
        return ierr;

    // calculate displacenents on a grid
    for (i = 0; i < uZ.nx; i++) {
        for (j = 0; j < uZ.ny; j++) {

            for (n = 0; n < nfault; n++) {
                ierr = fault[n].calculate(uZ.getX(i, j), uZ.getY(i, j), uzf);
                if (ierr)
                    return ierr;
                uZ(i, j) = uZ(i, j) + uzf;
            }
        }
    }

    return 0;
}

//=========================================================================
// Calculate surface displacements by summing effect from all ruptures
int cOkadaEarthquake::calculate(cOgrd &uZ, cOgrd &uLon, cOgrd &uLat)
{
    int    i, j, n, ierr;
    double uzf, ulonf, ulatf;

    if (!finalized)
        return Err.post("cOkadaEarthquake::calculate: eq not finalized");
    ierr = setGrid(uZ);
    if (ierr)
        return ierr;

    uLon = uZ;
    uLat = uZ;

    // calculate displacenents on a grid
    for (i = 0; i < uZ.nx; i++) {
        for (j = 0; j < uZ.ny; j++) {

            for (n = 0; n < nfault; n++) {
                ierr = fault[n].calculate(uZ.getX(i, j), uZ.getY(i, j), uzf, ulonf, ulatf);
                if (ierr)
                    return ierr;
                uZ(i, j)   = uZ(i, j) + uzf;
                uLon(i, j) = uLon(i, j) + ulonf;
                uLat(i, j) = uLat(i, j) + ulatf;
            }
        }
    }

    return 0;
}

//=========================================================================
// Check and prepare grid for calculations
int cOkadaEarthquake::setGrid(cOgrd &u)
{
    int ierr;

    if (u.xmin == u.xmax || u.ymin == u.ymax) {
        ierr = getDeformArea(1, u.xmin, u.xmax, u.ymin, u.ymax);
        if (ierr)
            return ierr;
    }

    if (u.nx * u.ny != 0 && u.dx * u.dy != 0)
        u.resetVal();
    else if (u.nx * u.ny != 0)
        u.initialize(u.xmin, u.xmax, u.nx, u.ymin, u.ymax, u.ny);
    else if (u.dx * u.dy != 0)
        u.initialize(u.xmin, u.xmax, u.dx, u.ymin, u.ymax, u.dy);
    else {
        u.dx = u.dy = 0.02; // a bit more than 1 arc minute (0.0166667)
        u.initialize(u.xmin, u.xmax, u.dx, u.ymin, u.ymax, u.dy);
    }

    return 0;
}

//=========================================================================
int cOkadaEarthquake::calculate(cObsArray &arr)
{
    int    k, n, ierr;
    double uzf, ulonf, ulatf;

    if (!finalized)
        return Err.post("cOkadaEarthquake::calculate: eq not finalized");

    for (k = 0; k < arr.nPos; k++) {
        for (arr.obs[k][0] = arr.obs[k][1] = arr.obs[k][2] = 0., n = 0; n < nfault; n++) {
            ierr = fault[n].calculate(arr.lon[k], arr.lat[k], uzf, ulonf, ulatf);
            if (ierr)
                return ierr;
            arr.obs[k][0] += ulonf;
            arr.obs[k][1] += ulatf;
            arr.obs[k][2] += uzf;
        }
    }

    return 0;
}

//=========================================================================
// print earthquake parameter data into a new string
char *cOkadaEarthquake::sprint()
{
    char *info, buf[256];
    int   n, bufsize;

    bufsize = 64 + nfault * 128;
    info    = new char[bufsize];

    sprintf(info, "Mw=%.2f M0=%-g", getMw(), getM0());
    sprintf(buf, "\nNumber of faults: %-3d", nfault);
    strcat(info, buf);
    for (n = 0; n < nfault; n++) {
        sprintf(buf, "\n  Fault %d(%d): length=%-.1f  width=%-.1f  slip=%-.2f  top=%-.2f", n + 1, nfault, fault[n].length / 1000, fault[n].width / 1000, fault[n].slip, fault[n].getZtop() / 1000);
        strcat(info, buf);
    }

    return info;
}
