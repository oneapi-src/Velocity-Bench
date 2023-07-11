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
#include <ctype.h>
#include <string.h>
#include "utilits.h"
#include "cSphere.h"
#include <cmath>

//=========================================================================
// Constructor
cObsArray::cObsArray()
{
    nPos = 0;
    nObs = 0;
    id   = NULL;
    lon  = NULL;
    lat  = NULL;
    obs  = NULL;
}

//=========================================================================
// Destructor
cObsArray::~cObsArray()
{
    if (lon)
        delete[] lon;
    if (lat)
        delete[] lat;
    if (obs) {
        for (int n = 0; n < nPos; n++)
            delete[] obs[n];
        delete[] obs;
    }
    if (id) {
        for (int n = 0; n < nPos; n++)
            delete[] id[n];
        delete[] id;
    }
}

//=========================================================================
// Reset observation array
int cObsArray::resetObs()
{
    if (!obs)
        return 0;

    for (int n = 0; n < nPos; n++) {
        memset(obs[n], 0, nObs * sizeof(double));
    }

    return 0;
}

//=========================================================================
// Fully reset observation array to a new number of observations
int cObsArray::resetObs(int newnobs)
{

    if (obs) {
        for (int n = 0; n < nPos; n++)
            delete[] obs[n];
        delete[] obs;
    }

    nObs = newnobs;
    obs  = new double *[nPos];
    for (int n = 0; n < nPos; n++) {
        obs[n] = new double[nObs];
        memset(obs[n], 0, nObs * sizeof(double));
    }

    return 0;
}

//=========================================================================
// Find poi
int cObsArray::findById(char *id0)
{
    int n;

    for (n = 0; n < nPos; n++)
        if (!strcmp(id0, id[n]))
            return n;

    return -1;
}

//=========================================================================
// Read observations from text file
int cObsArray::read(char *fname)
{
    FILE  *fp;
    char   record[256], buf[64];
    int    n, k, idPresent, line = 0;
    double lonr, latr;

    fp = fopen(fname, "rt");
    if (fp == NULL)
        return Err.post("Unable to open %s", fname);

    // Get number of values per location
    line = 0;
    if (utlReadNextRecord(fp, record, &line) == EOF) {
        if (fp != nullptr)
            fclose(fp);
        return Err.post("Unexpected EOF: %s", fname);
    }
    fclose(fp);
    // check if site ID's are available in the first column. Criterium: first character is not digit
    sscanf(record, "%s", buf);
    if (isalpha(buf[0]))
        idPresent = 1;
    else
        idPresent = 0;
    // nObs = number of columns minus 2 (lon lat) minus 1 (if idPresent)
    nObs = utlCountWordsInString(record) - 2 - idPresent;
    if (nObs < 0)
        return Err.post("%s:%d expected: [id lon lat [obs...]]", fname, line);

    // Get number of positions
    nPos = utlGetNumberOfRecords(fname);

    // Allocate memory
    lon = new double[nPos];
    lat = new double[nPos];
    if (idPresent) {
        id = new char *[nPos];
        for (n = 0; n < nPos; n++)
            id[n] = NULL;
    }
    if (nObs > 0) {
        obs = new double *[nPos];
        for (n = 0; n < nPos; n++) {
            obs[n] = new double[nObs];
            memset(obs[n], 0, nObs * sizeof(double));
        }
    }

    // Read data
    fp   = fopen(fname, "rt");
    line = 0;
    for (n = 0; n < nPos; n++) {
        if (utlReadNextRecord(fp, record, &line) == EOF) {
            fclose(fp);
            return Err.post("Unexpected EOF: %s", fname);
        }

        if (utlCountWordsInString(record) != (2 + idPresent + nObs)) {
            fclose(fp);
            return Err.post("%s:%d has invalid number of values", fname, line);
        }

        if (idPresent) {
            if (sscanf(record, "%s %lf %lf", buf, &lon[n], &lat[n]) != 3) {
                fclose(fp);
                return Err.post("%s:%d, expected id lon lat obs...", fname, line);
            }
            id[n] = strdup(buf);
        } else {
            if (sscanf(record, "%lf %lf", &lon[n], &lat[n]) != 2)
                return Err.post("%s:%d expected: lon lat obs...", fname, line);
        }

        for (k = 0; k < nObs; k++) {
            if (utlPickUpWordFromString(record, 3 + idPresent + k, buf) != 0) {
                fclose(fp);
                return Err.post("%s:%d expected: lon lat obs...", fname, line);
            }
            if (sscanf(buf, "%lf", &obs[n][k]) != 1) {
                fclose(fp);
                return Err.post("%s:%d expected: lon lat obs...", fname, line);
            }
        }
    }
    fclose(fp);

    return 0;
}

//=========================================================================
// Write to simple text file
int cObsArray::write(char *fname)
{
    FILE *fp;
    int   n, k;

    fp = fopen(fname, "wt");

    for (n = 0; n < nPos; n++) {

        if (id)
            fprintf(fp, "%s %g %g", id[n], lon[n], lat[n]);
        else
            fprintf(fp, "%g %g", lon[n], lat[n]);

        for (k = 0; k < nObs; k++)
            fprintf(fp, " %g", obs[n][k]);

        fprintf(fp, "\n");
    }

    fclose(fp);

    return 0;
}

//=========================================================================
// Write to binary stream
long cObsArray::writeBin(FILE *fp)
{
    long  bytes_written;
    float fbuf;

    bytes_written = 0;
    for (int n = 0; n < nPos; n++) {
        for (int k = 0; k < nObs; k++) {
            fbuf = (float)obs[n][k];
            fwrite(&fbuf, sizeof(float), 1, fp);
            bytes_written += sizeof(float);
        }
    }

    return bytes_written;
}

//=========================================================================
// Read from binary stream
long cObsArray::readBin(FILE *fp)
{
    long  bytes_read;
    float fbuf;

    bytes_read = 0;
    for (int n = 0; n < nPos; n++) {
        for (int k = 0; k < nObs; k++) {
            if (fread(&fbuf, sizeof(float), 1, fp) != 1)
                return utlPostError("Unexpected EOF");
            obs[n][k] = (double)fbuf;
            bytes_read += sizeof(float);
        }
    }

    return bytes_read;
}

//=========================================================================
// Calculate observation residual
double cObsArray::residual(cObsArray &ref)
{
    double resid = 0.;

    for (int n = 0; n < nPos; n++) {
        for (int k = 0; k < nObs; k++) {
            resid += pow((obs[n][k] - ref.obs[n][k]), 2.);
        }
    }

    resid = sqrt(resid) / nPos / nObs;

    return resid;
}

//=========================================================================
// Calculate norm of observations
double cObsArray::norm()
{
    double norm = 0.;

    for (int n = 0; n < nPos; n++) {
        for (int k = 0; k < nObs; k++) {
            norm += obs[n][k] * obs[n][k];
        }
    }

    norm = sqrt(norm) / nPos / nObs;

    return norm;
}

// Haversine formula for distance between any two points on the Earth surface
double GeoDistOnSphere(const double lon1, const double lat1, const double lon2, const double lat2)
{
    const double G2R    = 3.14159265358979 / 180.; // multiplyer to convert from degrees into radians
    const double REARTH = 6378.137;                // Earth radius in km along equator
    double       a, c, dist, rad;

    a = pow(sin(G2R * (lat2 - lat1) / 2), 2.) +
        cos(G2R * lat1) * cos(G2R * lat2) * pow(sin(G2R * (lon2 - lon1) / 2), 2.);
    rad = sqrt(a);
    if (rad > 1)
        rad = 1;
    c    = 2 * asin(rad);
    dist = REARTH * c;

    return dist;
}

double GeoStrikeOnSphere(double lon1, double lat1, double lon2, double lat2)
{
    const double G2R    = 3.14159265358979 / 180.; // multiplyer to convert from degrees into radians
    const double R2G    = 180. / 3.14159265358979; // multiplyer to convert from radians into degrees
    const double REARTH = 6378.137;                // Earth radius in km along equator
    double       strike, distRad;

    if ((lat1 == lat2) && (lon1 == lon2)) {
        strike = 0.;
    } else if (lon1 == lon2) {
        if (lat1 > lat2)
            strike = 180.;
        else
            strike = 0.;
    } else {
        distRad = GeoDistOnSphere(lon1, lat1, lon2, lat2) / REARTH;
        strike  = R2G * asin(cos(G2R * lat2) * sin(G2R * (lon2 - lon1)) / sin(distRad));

        if ((lat2 > lat1) && (lon2 > lon1)) {
        } else if ((lat2 < lat1) && (lon2 < lon1)) {
            strike = 180.0 - strike;
        } else if ((lat2 < lat1) && (lon2 > lon1)) {
            strike = 180.0 - strike;
        } else if ((lat2 > lat1) && (lon2 < lon1)) {
            strike += 360.0;
        }
    }

    //  if( strike > 180.0 ) strike -= 180.0;

    return strike;
}
