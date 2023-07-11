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
#include "easywave.h"
#include <cmath>

static char *IndexFile;
static int   Nrec2DOutput;

int ewStart2DOutput()
{
    FILE *fp;
    char  buf[64];

    // start index file
    sprintf(buf, "%s.2D.idx", Par.modelName);
    IndexFile = strdup(buf);

    fp = fopen(IndexFile, "wt");

    fprintf(fp, "%g %g %d %g %g %d\n", LonMin, LonMax, NLon, LatMin, LatMax, NLat);

    fclose(fp);

    Nrec2DOutput = 0;

    return 0;
}

int ewOut2D()
{
    FILE  *fp;
    short  nOutI, nOutJ;
    int    i, j, m;
    float  ftmp;
    double dtmp, lonOutMin, lonOutMax, latOutMin, latOutMax;
    char   record[128];

    CNode &Node = *gNode;

    Nrec2DOutput++;

    nOutI     = Imax - Imin + 1;
    lonOutMin = getLon(Imin);
    lonOutMax = getLon(Imax);
    nOutJ     = Jmax - Jmin + 1;
    latOutMin = getLat(Jmin);
    latOutMax = getLat(Jmax);

    // write ssh
    sprintf(record, "%s.2D.%5.5d.ssh", Par.modelName, Par.time);
    fp = fopen(record, "wb");
    fwrite("DSBB", 4, 1, fp);
    fwrite(&nOutI, sizeof(short), 1, fp);
    fwrite(&nOutJ, sizeof(short), 1, fp);
    fwrite(&lonOutMin, sizeof(double), 1, fp);
    fwrite(&lonOutMax, sizeof(double), 1, fp);
    fwrite(&latOutMin, sizeof(double), 1, fp);
    fwrite(&latOutMax, sizeof(double), 1, fp);
    dtmp = -1.;
    fwrite(&dtmp, sizeof(double), 1, fp);
    dtmp = +1.;
    fwrite(&dtmp, sizeof(double), 1, fp);
    for (j = Jmin; j <= Jmax; j++) {
        for (i = Imin; i <= Imax; i++) {
            m = idx(j, i);
            if (fabs(Node(m, iH)) < Par.sshTransparencyThreshold)
                ftmp = (float)9999;
            else
                ftmp = (float)Node(m, iH);
            fwrite(&ftmp, sizeof(float), 1, fp);
        }
    }
    fclose(fp);

    // updating contents file
    fp = fopen(IndexFile, "at");
    fprintf(fp, "%3.3d %s %d %d %d %d\n", Nrec2DOutput, utlTimeSplitString(Par.time), Imin, Imax, Jmin, Jmax);
    fclose(fp);

    return 0;
}

int ewDump2D()
{
    FILE  *fp;
    short  nOutI, nOutJ;
    int    i, j, m;
    float  ftmp, ftmpcalc;
    double dtmp, lonOutMin, lonOutMax, latOutMin, latOutMax;
    char   record[128];

    CNode &Node = *gNode;

    nOutI     = Imax - Imin + 1;
    lonOutMin = getLon(Imin);
    lonOutMax = getLon(Imax);
    nOutJ     = Jmax - Jmin + 1;
    latOutMin = getLat(Jmin);
    latOutMax = getLat(Jmax);

    // write ssh max
    sprintf(record, "%s.2D.sshmax", Par.modelName);
    fp = fopen(record, "wb");
    fwrite("DSBB", 4, 1, fp);
    fwrite(&nOutI, sizeof(short), 1, fp);
    fwrite(&nOutJ, sizeof(short), 1, fp);
    fwrite(&lonOutMin, sizeof(double), 1, fp);
    fwrite(&lonOutMax, sizeof(double), 1, fp);
    fwrite(&latOutMin, sizeof(double), 1, fp);
    fwrite(&latOutMax, sizeof(double), 1, fp);
    dtmp = 0.;
    fwrite(&dtmp, sizeof(double), 1, fp);
    dtmp = 1.;
    fwrite(&dtmp, sizeof(double), 1, fp);
    for (j = Jmin; j <= Jmax; j++) {
        for (i = Imin; i <= Imax; i++) {
            ftmp = (float)Node(idx(j, i), iHmax);
            fwrite(&ftmp, sizeof(float), 1, fp);
        }
    }
    fclose(fp);

    // write arrival times
    sprintf(record, "%s.2D.time", Par.modelName);
    fp = fopen(record, "wb");
    fwrite("DSBB", 4, 1, fp);
    fwrite(&nOutI, sizeof(short), 1, fp);
    fwrite(&nOutJ, sizeof(short), 1, fp);
    fwrite(&lonOutMin, sizeof(double), 1, fp);
    fwrite(&lonOutMax, sizeof(double), 1, fp);
    fwrite(&latOutMin, sizeof(double), 1, fp);
    fwrite(&latOutMax, sizeof(double), 1, fp);
    dtmp = 0.;
    fwrite(&dtmp, sizeof(double), 1, fp);
    dtmp = 1.;
    fwrite(&dtmp, sizeof(double), 1, fp);
    for (j = Jmin; j <= Jmax; j++) {
        for (i = Imin; i <= Imax; i++) {
            ftmp = (float)Node(idx(j, i), iTime);
            if (ftmp == -1.0f) {
                // no data value
                ftmpcalc = -1.0f;
            } else {
                ftmpcalc = ftmp / 60.0f;
            }
            fwrite(&ftmpcalc, sizeof(float), 1, fp);
        }
    }
    fclose(fp);

    return 0;
}
