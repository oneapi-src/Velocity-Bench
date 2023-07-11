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

#include "utilits.h"
#include "easywave.h"
#include <cmath>

//#define SSHMAX_TO_SINGLE_FILE 0

static int     MaxPOIs;
int            NPOIs;
static char  **idPOI;
long          *idxPOI;
static int    *flagRunupPOI;
static int     NtPOI;
static int    *timePOI;
static float **sshPOI;

int ewLoadPOIs()
{
    FILE  *fp, *fpFit, *fpAcc, *fpRej;
    int    ierr, line;
    int    i, j, i0, j0, imin, imax, jmin, jmax, flag, it, n;
    int    rad, nmin, itype;
    char   record[256], buf[256], id[64];
    double lon, lat, d2, d2min, lenLon, lenLat, depth;
    double POIdistMax, POIdepthMin, POIdepthMax;

    CNode &Node = *gNode;

    // if POIs file is not specified
    if (Par.filePOIs == NULL)
        return 0;

    Log.print("Loading POIs from %s", Par.filePOIs);

    MaxPOIs = utlGetNumberOfRecords(Par.filePOIs);
    if (!MaxPOIs)
        return Err.post("Empty POIs file");

    idPOI = new char *[MaxPOIs];
    if (!idPOI)
        return Err.post("Error allocating memory");
    idxPOI = new long[MaxPOIs];
    if (!idxPOI)
        return Err.post("Error allocating memory");
    flagRunupPOI = new int[MaxPOIs];
    if (!flagRunupPOI)
        return Err.post("Error allocating memory");
    sshPOI = new float *[MaxPOIs];
    if (!sshPOI)
        return Err.post("Error allocating memory");

    // read first record and get idea about the input type
    fp   = fopen(Par.filePOIs, "rt");
    line = 0;
    int const iReadError(utlReadNextRecord(fp, record, &line));
    itype = sscanf(record, "%s %s %s", buf, buf, buf);
    fclose(fp);

    if (itype == 2) { // poi-name and grid-index

        fp   = fopen(Par.filePOIs, "rt");
        line = NPOIs = 0;
        while (utlReadNextRecord(fp, record, &line) != EOF) {

            i = sscanf(record, "%s %d", id, &nmin);
            if (i != 2) {
                Log.print("! Bad POI record: %s", record);
                continue;
            }
            idPOI[NPOIs]        = strdup(id);
            idxPOI[NPOIs]       = nmin;
            flagRunupPOI[NPOIs] = 1;
            NPOIs++;
        }

        fclose(fp);
        Log.print("%d POIs of %d loaded successfully; %d POIs rejected", NPOIs, MaxPOIs, (MaxPOIs - NPOIs));
    } else if (itype == 3) { // poi-name and coordinates

        if (Par.poiReport) {
            fpAcc = fopen("poi_accepted.lst", "wt");
            fprintf(fpAcc, "ID lon lat   lonIJ latIJ depthIJ   dist[km]\n");
            fpRej = fopen("poi_rejected.lst", "wt");
        }

        lenLat = My_PI * Re / 180;

        fp   = fopen(Par.filePOIs, "rt");
        line = NPOIs = 0;
        while (utlReadNextRecord(fp, record, &line) != EOF) {

            i = sscanf(record, "%s %lf %lf %d", id, &lon, &lat, &flag);
            if (i == 3)
                flag = 1;
            else if (i == 4)
                ;
            else {
                Log.print("! Bad POI record: %s", record);
                if (Par.poiReport)
                    fprintf(fpRej, "%s\n", record);
                continue;
            }

            // find the closest water grid node. Local distances could be treated as cartesian (2 min cell distortion at 60 degrees is only about 2 meters or 0.2%)
            i0 = (int)((lon - LonMin) / DLon) + 1;
            j0 = (int)((lat - LatMin) / DLat) + 1;
            if (i0 < 1 || i0 > NLon || j0 < 1 || j0 > NLat) {
                Log.print("!POI out of grid: %s", record);
                if (Par.poiReport)
                    fprintf(fpRej, "%s\n", record);
                continue;
            }
            lenLon = lenLat * R6[j0];

            for (nmin = -1, rad = 0; rad < NLon && rad < NLat; rad++) {

                d2min = RealMax;

                imin = i0 - rad;
                if (imin < 1)
                    imin = 1;
                imax = i0 + rad + 1;
                if (imax > NLon)
                    imax = NLon;
                jmin = j0 - rad;
                if (jmin < 1)
                    jmin = 1;
                jmax = j0 + rad + 1;
                if (jmax > NLat)
                    jmax = NLat;
                for (i = imin; i <= imax; i++)
                    for (j = jmin; j <= jmax; j++) {
                        if (i != imin && i != imax && j != jmin && j != jmax)
                            continue;
                        n     = idx(j, i);
                        depth = Node(n, iD);
                        if (depth < Par.poiDepthMin || depth > Par.poiDepthMax)
                            continue;
                        d2 = pow(lenLon * (lon - getLon(i)), 2.) +
                             pow(lenLat * (lat - getLat(j)), 2.);
                        if (d2 < d2min) {
                            d2min = d2;
                            nmin  = n;
                        }
                    }

                if (nmin > 0)
                    break;
            }

            if (sqrt(d2min) > Par.poiDistMax) {
                Log.print("! Closest water node too far: %s", record);
                if (Par.poiReport)
                    fprintf(fpRej, "%s\n", record);
                continue;
            }

            idPOI[NPOIs]        = strdup(id);
            idxPOI[NPOIs]       = nmin;
            flagRunupPOI[NPOIs] = flag;
            NPOIs++;
            i = nmin / NLat + 1;
            j = nmin - (i - 1) * NLat + 1;
            if (Par.poiReport)
                fprintf(fpAcc, "%s %.4f %.4f   %.4f %.4f %.1f   %.3f\n", id, lon, lat, getLon(i), getLat(j), Node(nmin, iD), sqrt(d2min) / 1000);
        }

        fclose(fp);
        Log.print("%d POIs of %d loaded successfully; %d POIs rejected", NPOIs, MaxPOIs, (MaxPOIs - NPOIs));
        if (Par.poiReport) {
            fclose(fpAcc);
            fclose(fpRej);
        }
    }

    // if mareograms
    if (Par.poiDt) {
        NtPOI = Par.timeMax / Par.poiDt + 1;

        timePOI = new int[NtPOI];
        for (it = 0; it < NtPOI; it++)
            timePOI[it] = -1;

        for (n = 0; n < NPOIs; n++) {
            sshPOI[n] = new float[NtPOI];
            for (it = 0; it < NtPOI; it++)
                sshPOI[n][it] = 0.;
        }
    }

    return 0;
}

int ewSavePOIs()
{
    int    it, n;
    double ampFactor;

    CNode &Node = *gNode;

    if (!NPOIs)
        return 0;

    it = Par.time / Par.poiDt;

    timePOI[it] = Par.time;

    for (n = 0; n < NPOIs; n++) {

        if (flagRunupPOI[n])
            ampFactor = pow(Node(idxPOI[n], iD), 0.25);
        else
            ampFactor = 1.;

        sshPOI[n][it] = ampFactor * Node(idxPOI[n], iH);
    }

    return 0;
}

int ewDumpPOIs()
{
    FILE  *fp;
    char   buf[64];
    int    n, it;
    double ampFactor, dbuf;

    CNode &Node = *gNode;

    if (!NPOIs)
        return 0;

    if (Par.poiDt) { // Time series
        sprintf(buf, "%s.poi.ssh", Par.modelName);
        fp = fopen(buf, "wt");

        fprintf(fp, "Minute");
        for (n = 0; n < NPOIs; n++)
            fprintf(fp, "   %s", idPOI[n]);
        fprintf(fp, "\n");

        for (it = 0; (timePOI[it] != -1 && it < NtPOI); it++) {
            fprintf(fp, "%6.2f", (double)timePOI[it] / 60);
            for (n = 0; n < NPOIs; n++)
                fprintf(fp, " %7.3f", sshPOI[n][it]);
            fprintf(fp, "\n");
        }

        fclose(fp);
    }

    // EAT EWH
    sprintf(buf, "%s.poi.summary", Par.modelName);
    fp = fopen(buf, "wt");

    fprintf(fp, "ID ETA EWH\n");

    for (n = 0; n < NPOIs; n++) {
        fprintf(fp, "%s", idPOI[n]);
        dbuf = Node(idxPOI[n], iTime) / 60;
        if (dbuf < 0.)
            dbuf = -1.;
        fprintf(fp, " %6.2f", dbuf);

        if (flagRunupPOI[n])
            ampFactor = pow(Node(idxPOI[n], iD), 0.25);
        else
            ampFactor = 1;

        fprintf(fp, " %6.3f\n", (ampFactor * Node(idxPOI[n], iHmax)));
    }

    fclose(fp);

    return 0;
}
