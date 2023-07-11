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
#include "cOgrd.h"
#include "cOkadaEarthquake.h"
#include "easywave.h"
#include <cmath>
#include "FileHandler.h"
#include <chrono>

#define CPP_MODULE "SRCE"
#include "Logging.h"

int Imin;
int Imax;
int Jmin;
int Jmax;

#define SRC_GRD 1
#define SRC_FLT 2

//====================================================
int ewSource(double &dTotalIOReadTime)
{
    char             dsaa_label[8];
    int              i, j, ierr, srcType;
    double           lon, lat, dz, absuzmax, absuzmin;
    FILE            *fp;
    cOkadaEarthquake eq;
    cOgrd            uZ;
    double           dAccumulateIOTime(0.0);

    std::chrono::steady_clock::time_point tpStart;
    CNode                                &Node = *gNode;

    if (!Utility::FileHandler::FileExists(Par.fileSource))
        return 2;

    // check input file type: GRD or fault
    fp = fopen(Par.fileSource, "rb"); // Should not be nullptr
    LOG_ASSERT(fp != nullptr, "fopen failure");
    memset(dsaa_label, 0, 5);
    tpStart = std::chrono::steady_clock::now();
    ierr    = fread(dsaa_label, 4, 1, fp);
    dAccumulateIOTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - tpStart).count();
    if (!strcmp(dsaa_label, "DSAA") || !strcmp(dsaa_label, "DSBB"))
        srcType = SRC_GRD;
    else
        srcType = SRC_FLT;
    fclose(fp);

    // load GRD file
    if (srcType == SRC_GRD) {
        tpStart = std::chrono::steady_clock::now();
        ierr    = uZ.readGRD(Par.fileSource);
        dAccumulateIOTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - tpStart).count();
        if (ierr)
            return ierr;
    }

    // read fault(s) from file
    if (srcType == SRC_FLT) {
        int    effSymSource = 0;
        long   l;
        double dist, energy, factLat, effRad, effMax;

        tpStart = std::chrono::steady_clock::now();
        ierr    = eq.read(Par.fileSource);
        if (ierr)
            return ierr;
        dAccumulateIOTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - tpStart).count();

        if (Par.adjustZtop) {

            // check fault parameters
            Err.disable();
            ierr = eq.finalizeInput();
            while (ierr) {
                i    = ierr / 10;
                ierr = ierr - 10 * i;
                if (ierr == FLT_ERR_STRIKE) {
                    Log.print("No strike on input: Employing effective symmetric source model");
                    if (eq.nfault > 1) {
                        Err.enable();
                        return Err.post("Symmetric source assumes only 1 fault");
                    }
                    eq.fault[0].strike = 0.;
                    effSymSource       = 1;
                } else if (ierr == FLT_ERR_ZTOP) {
                    Log.print("Automatic depth correction to fault top @ 10 km");
                    eq.fault[i].depth = eq.fault[i].width / 2 * sindeg(eq.fault[i].dip) + 10.e3;
                } else {
                    Err.enable();
                    return ierr;
                }
                ierr = eq.finalizeInput();
            }
            Err.enable();

        } else {

            // check fault parameters
            Err.disable();
            ierr = eq.finalizeInput();
            if (ierr) {
                i    = ierr / 10;
                ierr = ierr - 10 * i;
                if (ierr != FLT_ERR_STRIKE) {
                    Err.enable();
                    ierr = eq.finalizeInput();
                    return ierr;
                }
                Log.print("No strike on input: Employing effective symmetric source model");
                Err.enable();
                if (eq.nfault > 1)
                    return Err.post("symmetric source assumes only 1 fault");
                eq.fault[0].strike = 0.;
                effSymSource       = 1;
                ierr               = eq.finalizeInput();
                if (ierr)
                    return ierr;
            }
            Err.enable();
        }

        // calculate uplift on a rectangular grid
        // set grid resolution, grid dimensions will be set automatically
        uZ.dx = DLon;
        uZ.dy = DLat;
        ierr  = eq.calculate(uZ);
        if (ierr)
            return ierr;

        if (effSymSource) {
            // integrate for tsunami energy
            energy = 0.;
            for (j = 0; j < uZ.ny; j++) {
                factLat = Dx * cosdeg(uZ.getY(0, j)) * Dy;
                for (i = 0; i < uZ.nx; i++)
                    energy += pow(uZ(i, j), 2.) * factLat;
            }
            energy *= (1000 * 9.81 / 2);
            effRad = eq.fault[0].length / sqrt(2 * M_PI);
            effMax = 1. / effRad / sqrt(M_PI / 2) / sqrt(1000 * 9.81 / 2) * sqrt(energy);
            Log.print("Effective source radius: %g km,  max height: %g m", effRad / 1000, effMax);

            // transfer uplift onto tsunami grid and define deformed area for acceleration
            for (i = 0; i < uZ.nx; i++) {
                for (j = 0; j < uZ.ny; j++) {
                    dist = GeoDistOnSphere(uZ.getX(i, j), uZ.getY(i, j), eq.fault[0].lon, eq.fault[0].lat) * 1000;
                    if (dist < effRad)
                        uZ(i, j) = effMax * cos(M_PI / 2 * dist / effRad);
                    else
                        uZ(i, j) = 0.;
                }
            }

        } // effective source

    } // src_type == fault

    dTotalIOReadTime += dAccumulateIOTime;
    // remove noise in the source
    absuzmax = uZ.getMaxAbsVal();

    if ((Par.ssh0ThresholdRel + Par.ssh0ThresholdAbs) != 0) {

        absuzmin = RealMax;
        if (Par.ssh0ThresholdRel != 0)
            absuzmin = Par.ssh0ThresholdRel * absuzmax;
        if (Par.ssh0ThresholdAbs != 0 && Par.ssh0ThresholdAbs < absuzmin)
            absuzmin = Par.ssh0ThresholdAbs;

        for (i = 0; i < uZ.nx; i++) {
            for (j = 0; j < uZ.ny; j++) {
                if (fabs(uZ(i, j)) < absuzmin)
                    uZ(i, j) = 0;
            }
        }
    }

    // calculated (if needed) arrival threshold (negative value means it is relative)
    if (Par.sshArrivalThreshold < 0)
        Par.sshArrivalThreshold = absuzmax * fabs(Par.sshArrivalThreshold);

    // transfer uplift onto tsunami grid and define deformed area for acceleration
    Imin = NLon;
    Imax = 1;
    Jmin = NLat;
    Jmax = 1;
    /* FIXME: change loops */
    for (i = 1; i <= NLon; i++) {
        for (j = 1; j <= NLat; j++) {

            lon = getLon(i);
            lat = getLat(j);

            if (Node(idx(j, i), iD) != 0.)
                dz = Node(idx(j, i), iH) = uZ.getVal(lon, lat);
            else
                dz = Node(idx(j, i), iH) = 0.;

            if (fabs(dz) > Par.sshClipThreshold) {
                Imin = My_min(Imin, i);
                Imax = My_max(Imax, i);
                Jmin = My_min(Jmin, j);
                Jmax = My_max(Jmax, j);
            }
        }
    }

    if (Imin == NLon)
        return Err.post("Zero initial displacement");

    Imin = My_max(Imin - 2, 2);
    Imax = My_min(Imax + 2, NLon - 1);
    Jmin = My_max(Jmin - 2, 2);
    Jmax = My_min(Jmax + 2, NLat - 1);

    return 0;
}
