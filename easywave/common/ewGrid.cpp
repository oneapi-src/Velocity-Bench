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
#include <stdlib.h>
#include <string.h>
#include <cassert>

#include "utilits.h"
#include "easywave.h"
#include <cmath>
#include "FileHandler.h"

#define CPP_MODULE "GRID"
#include "Logging.h"

int    NLon, NLat;
double LonMin, LonMax, LatMin, LatMax;
double DLon, DLat; // steps in grad
double Dx, Dy;     // steps in m, dx must be multiplied by cos(y) before use
float *R6;
float *C1;
float *C2;
float *C3;
float *C4;

int ewLoadBathymetry(double &dTotalIOReadTime)
{
    FILE          *fp;
    char           fileLabel[5];
    unsigned short shval;
    int            ierr, isBin, i, j, m, k;
    float          fval;
    double         dval;
    double         dAccumulateIOReadTime(0.0);

    std::chrono::steady_clock::time_point tpStart;
    CNode                                &Node = *gNode;

    Log.print("Loading bathymetry from %s", Par.fileBathymetry);

    if (!Utility::FileHandler::FileExists(Par.fileBathymetry)) {
        std::cerr << "Unable to find source file " << Par.fileBathymetry << std::endl;
        return 2;
    }

    // check if bathymetry file is in ascii or binary format

    fp = fopen(Par.fileBathymetry, "rb");
    assert(fp != nullptr); // Should never be empty once reached here

    memset(fileLabel, 0, 5);
    tpStart = std::chrono::steady_clock::now();
    ierr    = fread(fileLabel, 4, 1, fp);
    dAccumulateIOReadTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - tpStart).count();
    if (!strcmp(fileLabel, "DSAA"))
        isBin = 0;
    else if (!strcmp(fileLabel, "DSBB"))
        isBin = 1;
    else {
        if (fp != nullptr)
            fclose(fp);
        return Err.post("%s: not GRD-file!", Par.fileBathymetry);
    }

    fclose(fp);

    if (isBin) {
        tpStart = std::chrono::steady_clock::now();
        fp      = fopen(Par.fileBathymetry, "rb");
        ierr    = fread(fileLabel, 4, 1, fp);
        ierr    = fread(&shval, sizeof(unsigned short), 1, fp);
        NLon    = shval;
        ierr    = fread(&shval, sizeof(unsigned short), 1, fp);
        NLat    = shval;
        dAccumulateIOReadTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - tpStart).count();
    } else {
        tpStart = std::chrono::steady_clock::now();
        fp      = fopen(Par.fileBathymetry, "rt");
        ierr    = fscanf(fp, "%4s", fileLabel);
        ierr    = fscanf(fp, " %d %d ", &NLon, &NLat);
        dAccumulateIOReadTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - tpStart).count();
        if (NLon == -1 || NLat == -1) {
            fclose(fp);
            return Err.post("Data corrupted");
        }
    }

    // try to allocate memory for GRIDNODE structure and for caching arrays
    if (Node.mallocMem()) {
        fclose(fp);
        return Err.post("Error allocating memory");
    }

    if (isBin) {
        tpStart = std::chrono::steady_clock::now();
        ierr    = fread(&LonMin, sizeof(double), 1, fp);
        ierr    = fread(&LonMax, sizeof(double), 1, fp);
        ierr    = fread(&LatMin, sizeof(double), 1, fp);
        ierr    = fread(&LatMax, sizeof(double), 1, fp);
        ierr    = fread(&dval, sizeof(double), 1, fp);
        ierr    = fread(&dval, sizeof(double), 1, fp); // zmin zmax
        dAccumulateIOReadTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - tpStart).count();
    } else {
        tpStart = std::chrono::steady_clock::now();
        ierr    = fscanf(fp, " %lf %lf ", &LonMin, &LonMax);
        ierr    = fscanf(fp, " %lf %lf ", &LatMin, &LatMax);
        ierr    = fscanf(fp, " %*s %*s "); // zmin, zmax
        dAccumulateIOReadTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - tpStart).count();
    }

    if (NLon - 1 == 0) {
        fclose(fp);
        return Err.post("Fatal error, NLon - 1 is zero");
    }

    if (NLat - 1 == 0) {
        fclose(fp);
        return Err.post("Fatal error, NLat - 1 is zero");
    }

    DLon = (LonMax - LonMin) / (NLon - 1); // in degrees
    DLat = (LatMax - LatMin) / (NLat - 1);

    Dx = Re * g2r(DLon); // in m along the equator
    Dy = Re * g2r(DLat);

    if (isBin) {

        /* NOTE: optimal would be reading everything in one step, but that does not work because rows and columns are transposed
         * (only possible with binary data at all) - use temporary buffer for now (consumes additional memory!) */
        float *buf = new float[NLat * NLon];
        tpStart    = std::chrono::steady_clock::now();
        ierr       = fread(buf, sizeof(float), NLat * NLon, fp);
        dAccumulateIOReadTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - tpStart).count();

        for (i = 1; i <= NLon; i++) {
            for (j = 1; j <= NLat; j++) {

                m = idx(j, i);

                if (isBin)
                    fval = buf[(j - 1) * NLon + (i - 1)];
                // ierr = fread( &fval, sizeof(float), 1, fp );

                Node(m, iTopo) = fval;
                Node(m, iTime) = -1;
                Node(m, iD)    = -fval;

                if (Node(m, iD) < 0) {
                    Node(m, iD) = 0.0f;
                } else if (Node(m, iD) < Par.dmin) {
                    Node(m, iD) = Par.dmin;
                }
            }
        }

        delete[] buf;

    } else {

        for (j = 1; j <= NLat; j++) {
            for (i = 1; i <= NLon; i++) {

                m    = idx(j, i);
                ierr = fscanf(fp, " %f ", &fval);

                Node(m, iTopo) = fval;
                Node(m, iTime) = -1;
                Node(m, iD)    = -fval;

                if (Node(m, iD) < 0) {
                    Node(m, iD) = 0.0f;
                } else if (Node(m, iD) < Par.dmin) {
                    Node(m, iD) = Par.dmin;
                }
            }
        }
    }

    dTotalIOReadTime += dAccumulateIOReadTime;
    for (k = 1; k < MAX_VARS_PER_NODE - 2; k++) {
        Node.initMemory(k, 0);
    }

    fclose(fp);

    if (!Par.dt) { // time step not explicitly defined

        // Make bathymetry from topography. Compute stable time step.
        double dtLoc = RealMax;

        for (i = 1; i <= NLon; i++) {
            for (j = 1; j <= NLat; j++) {
                m = idx(j, i);
                if (Node(m, iD) == 0.0f)
                    continue;
                dtLoc = My_min(dtLoc, 0.8 * (Dx * cosdeg(getLat(j))) / sqrt(Gravity * Node(m, iD)));
            }
        }

        Log.print("Stable CFL time step: %g sec", dtLoc);
        if (dtLoc > 15)
            Par.dt = 15;
        else if (dtLoc > 10)
            Par.dt = 10;
        else if (dtLoc > 5)
            Par.dt = 5;
        else if (dtLoc > 2)
            Par.dt = 2;
        else if (dtLoc > 1)
            Par.dt = 1;
        else
            return Err.post("Bathymetry requires too small time step (<1sec)");
    }

    // Correct bathymetry for edge artefacts
    for (i = 1; i <= NLon; i++) {
        if (Node(idx(1, i), iD) != 0 && Node(idx(2, i), iD) == 0)
            Node(idx(1, i), iD) = 0.;
        if (Node(idx(NLat, i), iD) != 0 && Node(idx(NLat - 1, i), iD) == 0)
            Node(idx(NLat, i), iD) = 0.;
    }
    for (j = 1; j <= NLat; j++) {
        if (Node(idx(j, 1), iD) != 0 && Node(idx(j, 2), iD) == 0)
            Node(idx(j, 1), iD) = 0.;
        if (Node(idx(j, NLon), iD) != 0 && Node(idx(j, NLon - 1), iD) == 0)
            Node(idx(j, NLon), iD) = 0.;
    }

    // Calculate caching grid parameters for speedup
    for (j = 1; j <= NLat; j++) {
        R6[j] = cosdeg(LatMin + (j - 0.5) * DLat);
    }

    for (i = 1; i <= NLon; i++) {
        for (j = 1; j <= NLat; j++) {

            m = idx(j, i);

            if (Node(m, iD) == 0)
                continue;

            Node(m, iR1) = Par.dt / Dy / R6[j];

            if (i != NLon) {
                if (Node(m + NLat, iD) != 0) {
                    Node(m, iR2) = 0.5 * Gravity * Par.dt / Dy / R6[j] * (Node(m, iD) + Node(m + NLat, iD));
                    Node(m, iR3) = 0.5 * Par.dt * Omega * sindeg(LatMin + (j - 0.5) * DLat);
                }
            } else {
                Node(m, iR2) = 0.5 * Gravity * Par.dt / Dy / R6[j] * Node(m, iD) * 2;
                Node(m, iR3) = 0.5 * Par.dt * Omega * sindeg(LatMin + (j - 0.5) * DLat);
            }

            if (j != NLat) {
                if (Node(m + 1, iD) != 0) {
                    Node(m, iR4) = 0.5 * Gravity * Par.dt / Dy * (Node(m, iD) + Node(m + 1, iD));
                    Node(m, iR5) = 0.5 * Par.dt * Omega * sindeg(LatMin + j * DLat);
                }
            }
            /* FIXME: Bug? */
            else {
                Node(m, iR2) = 0.5 * Gravity * Par.dt / Dy * Node(m, iD) * 2;
                Node(m, iR3) = 0.5 * Par.dt * Omega * sindeg(LatMin + j * DLat);
            }
        }
    }

    for (i = 1; i <= NLon; i++) {
        C1[i] = 0;
        if (Node(idx(1, i), iD) != 0)
            C1[i] = 1. / sqrt(Gravity * Node(idx(1, i), iD));
        C3[i] = 0;
        if (Node(idx(NLat, i), iD) != 0)
            C3[i] = 1. / sqrt(Gravity * Node(idx(NLat, i), iD));
    }

    for (j = 1; j <= NLat; j++) {
        C2[j] = 0;
        if (Node(idx(j, 1), iD) != 0)
            C2[j] = 1. / sqrt(Gravity * Node(idx(j, 1), iD));
        C4[j] = 0;
        if (Node(idx(j, NLon), iD) != 0)
            C4[j] = 1. / sqrt(Gravity * Node(idx(j, NLon), iD));
    }

    return 0;
}
