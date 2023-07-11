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

#ifndef OGRD_H
#define OGRD_H

#include <stdio.h>

class cOgrd
{

  protected:
  public:
    int     nx, ny;
    int     nnod;
    double  noval;
    double  xmin, xmax;
    double  ymin, ymax;
    double  dx, dy;
    double *val;

    cOgrd();
    cOgrd(const cOgrd &);
    cOgrd(double xmin0, double xmax0, int nx0, double ymin0, double ymax0, int ny0);
    cOgrd(double xmin0, double xmax0, double dx0, double ymin0, double ymax0, double dy0);
    ~cOgrd();

    void    setNoval(double val);
    double  getNoval();
    int     initialize(double xmin0, double xmax0, int nx0, double ymin0, double ymax0, int ny0);
    int     initialize(double xmin0, double xmax0, double dx0, double ymin0, double ymax0, double dy0);
    int     readHeader(const char *grdfile);
    int     readGRD(const char *grdfile);
    int     readXYZ(const char *xyzfile);
    int     readRasterStream(FILE *fp, int ordering, int ydirection);
    cOgrd  *extract(int i1, int i2, int j1, int j2);
    cOgrd  *extract(double x1, double x2, double y1, double y2);
    int     idx(int i, int j);
    double &operator()(int i, int j);
    double &operator()(int idx);
    cOgrd  &operator=(const cOgrd &grd);
    cOgrd  &operator*=(double coeff);
    cOgrd  &operator+=(cOgrd &grd);
    void    getIJ(int idx, int &i, int &j);
    int     getIJ(double x, double y, int &i, int &j);
    double  getX(int i, int j);
    double  getX(int idx);
    double  getY(int i, int j);
    double  getY(int idx);
    double  getVal(int idx);
    double  getVal(int i, int j);
    double  getVal(double x, double y);
    double  getMaxVal();
    double  getMaxVal(int &i, int &j);
    double  getMinVal();
    double  getMinVal(int &i, int &j);
    double  getMaxAbsVal();
    double  getMaxAbsVal(int &i, int &j);
    double  getMaxAbsValBnd();
    void    setVal(double value, int i, int j);
    void    setVal(double value, int idx);
    void    reset();
    void    resetVal();
    int     getIntersectionRegion(const cOgrd &grd, int &imin, int &imax, int &jmin, int &jmax);
    int     interpolateFrom(cOgrd &grd, int resetValues);
    int     getNearestIdx(double x, double y);
    int     getNearestIdx(double x, double y, double rangemin, double rangemax);
    int     equalTo(cOgrd &grd);
    int     isSameShapeTo(cOgrd &grd);
    void    smooth(int radius);
    int     writeGRD(const char *fname);
    int     writeGRDbin(const char *fname);
    int     writeXYZ(const char *fname);
};

#endif // OGRD_H
