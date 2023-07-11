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
#include <math.h>
#include <iostream>

#include "utilits.h"
#include "cOgrd.h"

#define DEFAULT_NOVAL 0.0
#define ROWWISE       1
#define COLWISE       2
#define TOP2BOT       -1
#define BOT2TOP       1

int iabs(int i)
{
    return ((i < 0) ? (-i) : i);
}

//=========================================================================
// Zero Constructor
cOgrd::cOgrd()
{
    xmin = xmax = dx = 0.;
    nx               = 0;
    ymin = ymax = dy = 0.;
    ny               = 0;
    nnod             = 0;
    noval            = DEFAULT_NOVAL;
    val              = NULL;
}

//=========================================================================
// Copy constructor similar to operator =
cOgrd::cOgrd(const cOgrd &grd)
{
    xmin  = grd.xmin;
    xmax  = grd.xmax;
    dx    = grd.dx;
    nx    = grd.nx;
    ymin  = grd.ymin;
    ymax  = grd.ymax;
    dy    = grd.dy;
    ny    = grd.ny;
    nnod  = grd.nnod;
    noval = grd.noval;

    val = new double[nnod];
    memcpy(val, grd.val, nnod * sizeof(double));
}

//=========================================================================
cOgrd::cOgrd(double xmin0, double xmax0, int nx0, double ymin0, double ymax0, int ny0)
{
    noval = DEFAULT_NOVAL;
    val   = NULL;
    initialize(xmin0, xmax0, nx0, ymin0, ymax0, ny0);
}

//=========================================================================
cOgrd::cOgrd(double xmin0, double xmax0, double dx0, double ymin0, double ymax0, double dy0)
{
    noval = DEFAULT_NOVAL;
    val   = NULL;
    initialize(xmin0, xmax0, dx0, ymin0, ymax0, dy0);
}

//=========================================================================
// Destructor
cOgrd::~cOgrd()
{
    if (val)
        delete[] val;
}

//=========================================================================
// Initialize with number of nodes
int cOgrd::initialize(double xmin0, double xmax0, int nx0, double ymin0, double ymax0, int ny0)
{

    xmin = xmin0;
    xmax = xmax0;
    nx   = nx0;
    dx   = (xmax - xmin) / (nx - 1);

    ymin = ymin0;
    ymax = ymax0;
    ny   = ny0;
    dy   = (ymax - ymin) / (ny - 1);

    nnod = nx * ny;

    if (val)
        delete[] val;

    val = new double[nnod];
    for (int l = 0; l < nnod; l++)
        val[l] = noval;

    return 0;
}

//=========================================================================
// Initialize with grid step
int cOgrd::initialize(double xmin0, double xmax0, double dx0, double ymin0, double ymax0, double dy0)
{

    xmin = xmin0;
    xmax = xmax0;
    dx   = dx0;
    nx   = (int)((xmax - xmin) / dx + 1.e-10) + 1;

    ymin = ymin0;
    ymax = ymax0;
    dy   = dy0;
    ny   = (int)((ymax - ymin) / dy + 1.e-10) + 1;

    nnod = nx * ny;

    if (val)
        delete[] val;

    val = new double[nnod];
    for (int l = 0; l < nnod; l++)
        val[l] = noval;

    return 0;
}

//=========================================================================
void cOgrd::setNoval(double val)
{
    noval = val;
}

//=========================================================================
double cOgrd::getNoval()
{
    return noval;
}

//=========================================================================
// only read header from GRD-file
int cOgrd::readHeader(const char *grdfile)
{
    FILE  *fp;
    char   dsaa_label[8];
    int    ierr, isBin;
    short  shval;
    float  fval;
    double dval;

    if ((fp = fopen(grdfile, "rb")) == NULL)
        return Err.post("Unable to read file %s", grdfile);
    memset(dsaa_label, 0, 5);
    ierr = fread(dsaa_label, 4, 1, fp);
    if (!strcmp(dsaa_label, "DSAA"))
        isBin = 0;
    else if (!strcmp(dsaa_label, "DSBB"))
        isBin = 1;
    else {
        fclose(fp);
        return Err.post("%s is not a GRD-file", grdfile);
    }
    fclose(fp);

    if (isBin) {
        fp   = fopen(grdfile, "rb");
        ierr = fread(dsaa_label, 4, 1, fp);
        ierr = fread(&shval, sizeof(short), 1, fp);
        nx   = shval;
        ierr = fread(&shval, sizeof(short), 1, fp);
        ny   = shval;
        ierr = fread(&xmin, sizeof(double), 1, fp);
        ierr = fread(&xmax, sizeof(double), 1, fp);
        ierr = fread(&ymin, sizeof(double), 1, fp);
        ierr = fread(&ymax, sizeof(double), 1, fp);
        ierr = fread(&dval, sizeof(double), 1, fp);
        ierr = fread(&dval, sizeof(double), 1, fp); // zmin zmax
    } else {
        fp   = fopen(grdfile, "rt");
        ierr = fscanf(fp, "%7s", dsaa_label);
        ierr = fscanf(fp, " %d %d ", &nx, &ny);
        ierr = fscanf(fp, " %lf %lf ", &xmin, &xmax);
        ierr = fscanf(fp, " %lf %lf ", &ymin, &ymax);
        ierr = fscanf(fp, " %*s %*s "); // zmin, zmax
    }

    fclose(fp);

    nnod = nx * ny;

    if (nx < 2)
        return Err.post("nx - 1 is zero, aborting");

    if (ny < 2)
        return Err.post("ny - 1 is zero, aborting");

    dx = (xmax - xmin) / (nx - 1);
    dy = (ymax - ymin) / (ny - 1);

    return 0;
}

//=========================================================================
// Grid initialization from Golden Software GRD-file
int cOgrd::readGRD(const char *grdfile)
{
    FILE  *fp;
    char   dsaa_label[8];
    int    i, j, ierr, isBin;
    short  shval;
    float  fval;
    double dval;

    // check if bathymetry file is in ascii or binary format
    if ((fp = fopen(grdfile, "rb")) == NULL)
        return Err.post("Unable to read file %s", grdfile);
    memset(dsaa_label, 0, 5);
    ierr = fread(dsaa_label, 4, 1, fp);
    if (!strcmp(dsaa_label, "DSAA"))
        isBin = 0;
    else if (!strcmp(dsaa_label, "DSBB"))
        isBin = 1;
    else {
        fclose(fp);
        return Err.post("%s is not a GRD-file", grdfile);
    }
    fclose(fp);

    // Read Surfer GRD-file
    if (isBin) {
        fp   = fopen(grdfile, "rb");
        ierr = fread(dsaa_label, 4, 1, fp);
        ierr = fread(&shval, sizeof(short), 1, fp);
        nx   = shval;
        ierr = fread(&shval, sizeof(short), 1, fp);
        ny   = shval;
    } else {
        fp   = fopen(grdfile, "rt");
        ierr = fscanf(fp, "%7s", dsaa_label);
        ierr = fscanf(fp, " %d %d ", &nx, &ny);
    }

    nnod = nx * ny;

    if (val)
        delete[] val;
    val = new double[nnod];
    for (int l = 0; l < nnod; l++)
        val[l] = noval;

    if (isBin) {
        ierr = fread(&xmin, sizeof(double), 1, fp);
        ierr = fread(&xmax, sizeof(double), 1, fp);
        ierr = fread(&ymin, sizeof(double), 1, fp);
        ierr = fread(&ymax, sizeof(double), 1, fp);
        ierr = fread(&dval, sizeof(double), 1, fp);
        ierr = fread(&dval, sizeof(double), 1, fp); // zmin zmax
    } else {
        ierr = fscanf(fp, " %lf %lf ", &xmin, &xmax);
        ierr = fscanf(fp, " %lf %lf ", &ymin, &ymax);
        ierr = fscanf(fp, " %*s %*s "); // zmin, zmax
    }

    dx = (xmax - xmin) / (nx - 1);
    dy = (ymax - ymin) / (ny - 1);

    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {

            if (isBin)
                ierr = fread(&fval, sizeof(float), 1, fp);
            else
                ierr = fscanf(fp, " %f ", &fval);

            val[idx(i, j)] = (double)fval;
        }
    }

    fclose(fp);

    return 0;
}

//=========================================================================
// Grid initialization from XYZ-file
int cOgrd::readXYZ(const char *fname)
{
#define MAXRECLEN 254
    FILE  *fp;
    char   record[254];
    int    i, j, line;
    int    ordering(0), ydirection(0);
    int    nxny;
    double x0, x, y0, y, z;

    // Open plain XYZ-file
    if ((fp = fopen(fname, "rt")) == NULL) {
        return utlPostError("Unable to open file");
    }

    // Check xyz-file format, define number of nodes and min-max
    rewind(fp);
    for (line = 0, nxny = 0, xmin = ymin = RealMax, xmax = ymax = -RealMax; utlReadNextDataRecord(fp, record, &line) != EOF;) {
        if (sscanf(record, "%lf %lf %lf", &x, &y, &z) != 3) {
            if (fp != nullptr)
                fclose(fp);
            return Err.post("%s:%d has an X Y Z error", fname, line);
        }
        nxny++;
        if (x < xmin)
            xmin = x;
        if (x > xmax)
            xmax = x;
        if (y < ymin)
            ymin = y;
        if (y > ymax)
            ymax = y;
    }

    // Read first two lines and define if file is row- or column-wise
    rewind(fp);
    line = 0;
    utlReadNextDataRecord(fp, record, &line);
    sscanf(record, "%lf %lf %lf", &x0, &y0, &z);
    utlReadNextDataRecord(fp, record, &line);
    sscanf(record, "%lf %lf %lf", &x, &y, &z);
    if (x0 == x && y0 != y)
        ordering = COLWISE;
    else if (x0 != x && y0 == y)
        ordering = ROWWISE;
    else {
        if (fp != nullptr)
            fclose(fp);
        return Err.post("%s:%d has a data error", fname, line);
    }

    // Define nx and ny
    rewind(fp);
    line = 0;
    utlReadNextDataRecord(fp, record, &line);
    sscanf(record, "%lf %lf %lf", &x0, &y0, &z);
    if (ordering == ROWWISE) {
        nx = 1;
        while (utlReadNextDataRecord(fp, record, &line) != EOF) {
            sscanf(record, "%lf %lf %lf", &x, &y, &z);
            if (y != y0)
                break;
            nx++;
        }
        ny = nxny / nx;
        if (y > y0)
            ydirection = BOT2TOP;
        else
            ydirection = TOP2BOT;

    } else if (ordering == COLWISE) {
        ny = 1;
        while (utlReadNextDataRecord(fp, record, &line) != EOF) {
            sscanf(record, "%lf %lf %lf", &x, &y, &z);
            if (x != x0)
                break;
            if (ny == 1) {
                if (y > y0)
                    ydirection = BOT2TOP;
                else
                    ydirection = TOP2BOT;
            }
            ny++;
        }
        nx = nxny / ny;
    }

    if (nx - 1 == 0) {
        if (fp != nullptr)
            fclose(fp);
        return Err.post("nx - 1 is zero, potential division by 0");
    }

    if (ny - 1 == 0) {
        if (fp != nullptr)
            fclose(fp);
        return Err.post("ny - 1 is zero, potential division by 0");
    }

    if (nx * ny != nxny) {
        if (fp != nullptr)
            fclose(fp);
        return Err.post("cOgrd::readXYZ -> nx*ny != nxny");
    }

    // Other grid parameters
    dx = (xmax - xmin) / (nx - 1);
    dy = (ymax - ymin) / (ny - 1);

    nnod = nx * ny;

    // Allocate memory for z-values
    if (val)
        delete[] val;
    val = new double[nnod];
    for (int l = 0; l < nnod; l++)
        val[l] = noval;

    // Read z-values
    rewind(fp);
    line = 0;
    if (ordering == ROWWISE) {
        if (ydirection == BOT2TOP) {
            for (j = 0; j < ny; j++) {
                for (i = 0; i < nx; i++) {
                    utlReadNextDataRecord(fp, record, &line);
                    sscanf(record, "%*s %*s %lf", &val[idx(i, j)]);
                }
            }
        } else if (ydirection == TOP2BOT) {
            for (j = ny - 1; j >= 0; j--) {
                for (i = 0; i < nx; i++) {
                    utlReadNextDataRecord(fp, record, &line);
                    sscanf(record, "%*s %*s %lf", &val[idx(i, j)]);
                }
            }
        }
    } else if (ordering == COLWISE) {
        if (ydirection == BOT2TOP) {
            for (i = 0; i < nx; i++) {
                for (j = 0; j < ny; j++) {
                    utlReadNextDataRecord(fp, record, &line);
                    sscanf(record, "%*s %*s %lf", &val[idx(i, j)]);
                }
            }
        } else if (ydirection == TOP2BOT) {
            for (i = 0; i < nx; i++) {
                for (j = ny - 1; j >= 0; j--) {
                    utlReadNextDataRecord(fp, record, &line);
                    sscanf(record, "%*s %*s %lf", &val[idx(i, j)]);
                }
            }
        }
    }

    fclose(fp);

    return 0;
}

//=========================================================================
// Read grid from ASCII-raster file
int cOgrd::readRasterStream(FILE *fp, int ordering, int ydirection)
{
    int    i, j;
    double x0, x, y0, y, z;

    if (ordering == ROWWISE) {
        if (ydirection == BOT2TOP) {
            for (j = 0; j < ny; j++)
                for (i = 0; i < nx; i++)
                    if (fscanf(fp, "%lf", &val[idx(i, j)]) != 1)
                        return Err.post("Unexpected EOF");
        } else if (ydirection == TOP2BOT) {
            for (j = ny - 1; j >= 0; j--)
                for (i = 0; i < nx; i++)
                    if (fscanf(fp, "%lf", &val[idx(i, j)]) != 1)
                        return Err.post("Unexpected EOF");
        } else
            return Err.post("Unexpected direction");
    } else if (ordering == COLWISE) {
        if (ydirection == BOT2TOP) {
            for (i = 0; i < nx; i++)
                for (j = 0; j < ny; j++)
                    if (fscanf(fp, "%lf", &val[idx(i, j)]) != 1)
                        return Err.post("Unexpected EOF");
        } else if (ydirection == TOP2BOT) {
            for (i = 0; i < nx; i++)
                for (j = ny - 1; j >= 0; j--)
                    if (fscanf(fp, "%lf", &val[idx(i, j)]) != 1)
                        return Err.post("Unexpected EOF");
        } else
            return Err.post("Unexpected direction");
    } else
        return Err.post("Unexpected Ordering");

    return 0;
}

//=========================================================================
// Cut-off a subgrid
cOgrd *cOgrd::extract(int i1, int i2, int j1, int j2)
{
    cOgrd *grd;

    if (i1 < 0 || i2 > (nx - 1) || j1 < 0 || j2 > (ny - 1)) {
        Err.post("subGrid: bad ranges");
        return NULL;
    }

    grd = new cOgrd(getX(i1, 0), getX(i2, 0), (i2 - i1 + 1), getY(0, j1), getY(0, j2), (j2 - j1 + 1));

    for (int i = i1; i <= i2; i++) {
        for (int j = j1; j <= j2; j++) {
            (*grd).setVal(getVal(i, j), (i - i1), (j - j1));
        }
    }

    return grd;
}

//=========================================================================
// Cut-off a subgrid
cOgrd *cOgrd::extract(double x1, double x2, double y1, double y2)
{
    int i1, i2, j1, j2;

    i1 = (int)((x1 - xmin) / dx);
    if (i1 < 0)
        i1 = 0;
    i2 = (int)((x2 - xmin) / dx);
    if (i2 > (nx - 1))
        i2 = nx - 1;
    j1 = (int)((y1 - ymin) / dy);
    if (j1 < 0)
        j1 = 0;
    j2 = (int)((y2 - ymin) / dy);
    if (j2 > (ny - 1))
        j2 = ny - 1;

    return extract(i1, i2, j1, j2);
}

//=========================================================================
// Total reset
void cOgrd::reset()
{
    xmin = xmax = dx = 0.;
    nx               = 0;
    ymin = ymax = dy = 0.;
    ny               = 0;

    nnod  = 0;
    noval = DEFAULT_NOVAL;

    if (val != NULL)
        delete[] val;
    val = NULL;
}

//=========================================================================
// Reset values to zero
void cOgrd::resetVal()
{
    for (int l = 0; l < nnod; l++)
        val[l] = noval;
}

//=========================================================================
cOgrd &cOgrd::operator=(const cOgrd &grd)
{
    if (this == &grd)
        return *this;

    xmin = grd.xmin;
    xmax = grd.xmax;
    dx   = grd.dx;
    nx   = grd.nx;
    ymin = grd.ymin;
    ymax = grd.ymax;
    dy   = grd.dy;
    ny   = grd.ny;
    nnod = grd.nnod;

    if (val)
        delete[] val;
    val = new double[nnod];
    memcpy(val, grd.val, nnod * sizeof(double));

    return *this;
}

//=========================================================================
cOgrd &cOgrd::operator*=(double coeff)
{

    for (int l = 0; l < nnod; l++)
        val[l] *= coeff;

    return *this;
}

//=========================================================================
cOgrd &cOgrd::operator+=(cOgrd &grd)
{
    int l;
    int i, j, i1, i2, j1, j2;

    // if the two grids have the same shape
    if (xmin == grd.xmin && xmax == grd.xmax && nx == grd.nx && ymin == grd.ymin && ymax == grd.ymax && ny == grd.ny) {

        for (l = 0; l < nnod; l++)
            val[l] += grd.val[l];
    } else { // the two grids have different shapes. Do bi-linear interpolation

        // straightforward but somewhat slow variant
        // for( l=0; l<nnod; l++ )
        // val[l] += grd.getVal( getX(l), getY(l) );

        // accelerate: use only the actual part of the grid
        i1 = (int)((grd.xmin - xmin) / dx);
        if (i1 < 0)
            i1 = 0;
        i2 = (int)((grd.xmax - xmin) / dx) + 1;
        if (i2 > nx - 1)
            i2 = nx - 1;
        j1 = (int)((grd.ymin - ymin) / dy);
        if (j1 < 0)
            j1 = 0;
        j2 = (int)((grd.ymax - ymin) / dy) + 1;
        if (j2 > ny - 1)
            j2 = ny - 1;
        for (i = i1; i <= i2; i++) {
            for (j = j1; j <= j2; j++) {
                val[idx(i, j)] += grd.getVal(getX(i, j), getY(i, j));
            }
        }
    }

    return *this;
}

//=========================================================================
double &cOgrd::operator()(int i, int j)
{
    return (val[idx(i, j)]);
}

//=========================================================================
double &cOgrd::operator()(int l)
{
    return (val[l]);
}

//=========================================================================
// Get IJ-indices from the offset
void cOgrd::getIJ(int idx, int &i, int &j)
{

    // J is the inner loop (increments faster than I)
    i = idx / ny;
    j = idx - i * ny;
}

//=========================================================================
// Get IJ-indices from coordinates
int cOgrd::getIJ(double x, double y, int &i, int &j)
{
    i = (int)((x - xmin) / (xmax - xmin) * nx);
    if (i < 0 || i > (nx - 1))
        return -1;
    j = (int)((y - ymin) / (ymax - ymin) * ny);
    if (j < 0 || j > (ny - 1))
        return -1;

    return 0;
}

//=========================================================================
// Get offset from IJ-indices
int cOgrd::idx(int i, int j)
{

    // J is the inner loop (increments faster than I)
    return (int((int)ny * i + j));
}

//=========================================================================
// Get value at given node
double cOgrd::getVal(int i, int j)
{

    return (val[idx(i, j)]);
}

//=========================================================================
// Get value at given node
double cOgrd::getVal(int idx)
{

    return (val[idx]);
}

//=========================================================================
// Set value at given node
void cOgrd::setVal(double value, int i, int j)
{

    val[idx(i, j)] = value;
}

//=========================================================================
// Set value at given node
void cOgrd::setVal(double value, int idx)
{

    val[idx] = value;
}

//=========================================================================
// Get maximal value on a grid
double cOgrd::getMaxVal()
{
    int    l;
    double vmax;

    for (vmax = -RealMax, l = 0; l < nnod; l++)
        if (val[l] > vmax)
            vmax = val[l];

    return vmax;
}

//=========================================================================
// Get maximal value and its position on a grid
double cOgrd::getMaxVal(int &i, int &j)
{
    int    l, lmax;
    double vmax;

    for (lmax = 0, vmax = -RealMax, l = 0; l < nnod; l++) {
        if (val[l] > vmax) {
            vmax = val[l];
            lmax = l;
        }
    }

    getIJ(lmax, i, j);

    return vmax;
}

//=========================================================================
// Get minimal value on a grid
double cOgrd::getMinVal()
{
    int    l;
    double vmin;

    for (vmin = RealMax, l = 0; l < nnod; l++)
        if (val[l] < vmin)
            vmin = val[l];

    return vmin;
}

//=========================================================================
// Get minimal value and its position on a grid
double cOgrd::getMinVal(int &i, int &j)
{
    int    l, lmin;
    double vmin;

    for (lmin = 0, vmin = RealMax, l = 0; l < nnod; l++) {
        if (val[l] < vmin) {
            vmin = val[l];
            lmin = l;
        }
    }

    getIJ(lmin, i, j);

    return vmin;
}

//=========================================================================
// Get maximal absolute value on a grid
double cOgrd::getMaxAbsVal()
{
    int    l;
    double vmax;

    for (vmax = -RealMax, l = 0; l < nnod; l++)
        if (fabs(val[l]) > vmax)
            vmax = fabs(val[l]);

    return vmax;
}

//=========================================================================
// Get maximal absolute value and its position on a grid
double cOgrd::getMaxAbsVal(int &i, int &j)
{
    int    l, lmax;
    double vmax;

    for (lmax = 0, vmax = -RealMax, l = 0; l < nnod; l++) {
        if (fabs(val[l]) > vmax) {
            vmax = fabs(val[l]);
            lmax = l;
        }
    }

    getIJ(lmax, i, j);

    return vmax;
}

//=========================================================================
// Get maximal absolute value along grid boundaries
double cOgrd::getMaxAbsValBnd()
{
    int    i, j;
    double vmax = -RealMax;

    for (i = 0; i < nx; i++) {
        if (fabs(val[idx(i, 0)]) > vmax)
            vmax = fabs(val[idx(i, 0)]);
        if (fabs(val[idx(i, ny - 1)]) > vmax)
            vmax = fabs(val[idx(i, ny - 1)]);
    }

    for (j = 0; j < ny; j++) {
        if (fabs(val[idx(0, j)]) > vmax)
            vmax = fabs(val[idx(0, j)]);
        if (fabs(val[idx(nx - 1, j)]) > vmax)
            vmax = fabs(val[idx(nx - 1, j)]);
    }

    return vmax;
}

//=========================================================================
// Get value with interpolation (bi-linear)
double cOgrd::getVal(double x, double y)
{
    int    i0, j0;
    double fx, fy, val_l, val_r, result;

    i0 = (int)((x - xmin) / dx);
    if (i0 < 0 || (i0 + 1) >= nx)
        return (noval);
    j0 = (int)((y - ymin) / dy);
    if (j0 < 0 || (j0 + 1) >= ny)
        return (noval);

    fx = (x - (xmin + dx * i0)) / dx;
    fy = (y - (ymin + dy * j0)) / dy;

    val_l = (1 - fy) * getVal(i0, j0) + fy * getVal(i0, j0 + 1);
    val_r = (1 - fy) * getVal(i0 + 1, j0) + fy * getVal(i0 + 1, j0 + 1);

    result = (1 - fx) * val_l + fx * val_r;

    return (result);
}

//=========================================================================
// Get longitude from IJ-indeces
double cOgrd::getX(int i, int /*j*/)
{
    return (xmin + i * dx);
}

//=========================================================================
// Get longitude from the offset
double cOgrd::getX(int idx)
{
    int i, j;

    getIJ(idx, i, j);
    return (xmin + i * dx);
}

//=========================================================================
// Get lattitude from IJ-indeces
double cOgrd::getY(int /*i*/, int j)
{
    return (ymin + j * dy);
}

//=========================================================================
// Get lattitude from the offset
double cOgrd::getY(int idx)
{
    int i, j;

    getIJ(idx, i, j);
    return (ymin + j * dy);
}

//=========================================================================
// Check if the shapes are equal
int cOgrd::isSameShapeTo(cOgrd &grd)
{
    if (xmin != grd.xmin)
        return 0;
    if (xmax != grd.xmax)
        return 0;
    if (nx != grd.nx)
        return 0;
    if (ymin != grd.ymin)
        return 0;
    if (ymax != grd.ymax)
        return 0;
    if (ny != grd.ny)
        return 0;

    return 1;
}

//=========================================================================
// INtersection region with another grid
int cOgrd::getIntersectionRegion(const cOgrd &grd, int &imin, int &imax, int &jmin, int &jmax)
{

    if (xmin < grd.xmin) {
        if (xmax < grd.xmin)
            return -1;
        imin = (int)((grd.xmin - xmin) / dx);
    } else if (xmin <= grd.xmax) {
        imin = 0;
    } else
        return -1;

    if (xmax < grd.xmin)
        return -1;
    else if (xmax <= grd.xmax) {
        imax = nx - 1;
    } else {
        imax = (int)((grd.xmax - xmin) / dx);
    }

    if (ymin < grd.ymin) {
        if (ymax < grd.ymin)
            return -1;
        jmin = (int)((grd.ymin - ymin) / dy);
    } else if (ymin <= grd.ymax) {
        jmin = 0;
    } else
        return -1;

    if (ymax < grd.ymin)
        return -1;
    else if (ymax <= grd.ymax) {
        jmax = ny - 1;
    } else {
        jmax = (int)((grd.ymax - ymin) / dy);
    }

    return 0;
}

//=========================================================================
// Interpolate grid values from another O-grid (bi-linear)
int cOgrd::interpolateFrom(cOgrd &grd, int resetValues)
{
    int    ierr, imin, imax, jmin, jmax;
    double value;

    if (resetValues)
        resetVal();

    ierr = getIntersectionRegion(grd, imin, imax, jmin, jmax);
    if (ierr)
        return 0;

    for (int i = imin; i <= imax; i++) {
        for (int j = jmin; j <= jmax; j++) {
            value = grd.getVal(getX(i, j), getY(i, j));
            if (value != grd.noval)
                val[idx(i, j)] = value;
        }
    }

    return (0);
}

//=========================================================================
// Get nearest node
int cOgrd::getNearestIdx(double x0, double y0)
{
    int    i0, j0;
    int    l, lmin;
    double dist2, dist2min;

    if (x0 < xmin || x0 > xmax || y0 < ymin || y0 > ymax)
        return -1;

    i0 = (int)((x0 - xmin) / dx);
    j0 = (int)((y0 - ymin) / dy);

    l        = idx(i0, j0);
    dist2min = (x0 - getX(l)) * (x0 - getX(l)) + (y0 - getY(l)) * (y0 - getY(l));
    lmin     = l;

    l     = idx(i0 + 1, j0);
    dist2 = (x0 - getX(l)) * (x0 - getX(l)) + (y0 - getY(l)) * (y0 - getY(l));
    if (dist2 < dist2min) {
        dist2min = dist2;
        lmin     = l;
    }

    l     = idx(i0, j0 + 1);
    dist2 = (x0 - getX(l)) * (x0 - getX(l)) + (y0 - getY(l)) * (y0 - getY(l));
    if (dist2 < dist2min) {
        dist2min = dist2;
        lmin     = l;
    }

    l     = idx(i0 + 1, j0 + 1);
    dist2 = (x0 - getX(l)) * (x0 - getX(l)) + (y0 - getY(l)) * (y0 - getY(l));
    if (dist2 < dist2min) {
        dist2min = dist2;
        lmin     = l;
    }

    return lmin;
}

//=========================================================================
// Get nearest conditioned node
int cOgrd::getNearestIdx(double x0, double y0, double rangemin, double rangemax)
{
    int    i, j, i0, j0, rad;
    int    l, lmin;
    double dist2, dist2min;

    lmin = getNearestIdx(x0, y0);
    if (lmin == -1)
        return lmin;

    if (val[lmin] >= rangemin && val[lmin] <= rangemax)
        return lmin;

    getIJ(lmin, i0, j0);

    for (lmin = -1, rad = 1; rad < nx && rad < ny; rad++) {

        dist2min = RealMax;

        for (i = i0 - rad; i <= i0 + rad; i++)
            for (j = j0 - rad; j <= j0 + rad; j++) {
                if (iabs(i - i0) != rad && iabs(j - j0) != rad)
                    continue;
                if (i < 0 || i > nx - 1 || j < 0 || j > ny - 1)
                    continue;
                l = idx(i, j);
                if (val[l] < rangemin || val[l] > rangemax)
                    continue;
                dist2 = (x0 - getX(l)) * (x0 - getX(l)) + (y0 - getY(l)) * (y0 - getY(l));
                if (dist2 < dist2min) {
                    dist2min = dist2;
                    lmin     = l;
                }
            }

        if (lmin > 0)
            break;
    }

    return lmin;
}

//=========================================================================
// Smooth data on grid
void cOgrd::smooth(int radius)
{
    int    i, j, k, ik, jk;
    int    l;
    double sumwt;
    double wt[10] = {1.0, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};

    if (radius == 0)
        return;

    cOgrd tmp(*this);

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {

            l = idx(i, j);

            for (sumwt = tmp(l) = 0, k = 0; k <= radius; k++) {
                for (ik = i - k; ik <= i + k; ik++) {
                    if (ik < 0 || ik >= nx)
                        continue;
                    for (jk = j - k; jk <= j + k; jk++) {
                        if (jk < 0 || jk >= ny)
                            continue;
                        if (iabs(ik - i) == k || iabs(jk - j) == k) {
                            tmp(l) += wt[k] * ((*this)(ik, jk));
                            sumwt += wt[k];
                        }
                    }
                }
            }
            tmp(l) /= sumwt;
        }
    }

    *this = tmp;
}

//=========================================================================
// write to GRD-file
int cOgrd::writeGRD(const char *fname)
{
    FILE *fp;
    int   i, j, cnt;

    fp = fopen(fname, "wt");

    fprintf(fp, "DSAA\n");
    fprintf(fp, "%d %d\n", nx, ny);
    fprintf(fp, "%f %f\n", xmin, xmax);
    fprintf(fp, "%f %f\n", ymin, ymax);
    fprintf(fp, "%f %f\n", getMinVal(), getMaxVal());

    for (cnt = 0, j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            cnt++;
            fprintf(fp, " %g", val[idx(i, j)]);
            if (cnt == 10) {
                fprintf(fp, "\n");
                cnt = 0;
            }
        }
        fprintf(fp, "\n\n");
        cnt = 0;
    }

    fclose(fp);

    return 0;
}

//=========================================================================
// write to GRD-file: binary version
int cOgrd::writeGRDbin(const char *fname)
{
    FILE  *fp;
    short  i2buf;
    float  r4buf;
    double r8buf;
    int    i, j;

    fp = fopen(fname, "wb");

    fwrite("DSBB", 4, 1, fp);
    i2buf = (short)nx;
    fwrite(&i2buf, sizeof(short), 1, fp);
    i2buf = (short)ny;
    fwrite(&i2buf, sizeof(short), 1, fp);
    fwrite(&xmin, sizeof(double), 1, fp);
    fwrite(&xmax, sizeof(double), 1, fp);
    fwrite(&ymin, sizeof(double), 1, fp);
    fwrite(&ymax, sizeof(double), 1, fp);
    r8buf = (double)getMinVal();
    fwrite(&r8buf, sizeof(double), 1, fp);
    r8buf = (double)getMaxVal();
    fwrite(&r8buf, sizeof(double), 1, fp);

    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            r4buf = (float)val[idx(i, j)];
            fwrite(&r4buf, sizeof(float), 1, fp);
        }
    }

    fclose(fp);

    return 0;
}

//=========================================================================
// write in ascii 3 column-format
int cOgrd::writeXYZ(const char *xyzfile)
{
    FILE *fp;
    int   i, j, l;

    fp = fopen(xyzfile, "wt");

    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            l = idx(i, j);
            fprintf(fp, " %g %g %g\n", getX(l), getY(l), val[l]);
        }
    }

    fclose(fp);

    return 0;
}
