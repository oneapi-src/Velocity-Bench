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

#ifndef OKADAFAULT_H
#define OKADAFAULT_H

// Position of the fault reference point (to which lon/lat/depth are related)
#define FLT_POS_C  0 // center of the fault, Okada's coordinates: (L/2;W/2)
#define FLT_POS_MT 1 // middle of the top edge: (L/2;W)
#define FLT_POS_BT 2 // beginning of the top edge: (0;W)
#define FLT_POS_BB 3 // beginning of the bottom edge: (0;0)
#define FLT_POS_MB 4 // middle of the bottom edge: (L/2;0)

#define FLT_ERR_DATA     1
#define FLT_ERR_ZTOP     3
#define FLT_ERR_INTERNAL 4
#define FLT_ERR_STRIKE   6

class cOkadaFault
{

  protected:
    int    checked;
    int    adjust_depth;
    double sind;
    double cosd;
    double sins;
    double coss;
    double tand;
    double coslat;
    double zbot;
    double wp;
    double dslip;
    double sslip;

    double mw2m0();

  public:
    int    refpos; // 0- one of POS_XX (see above definitions)
    double mw;
    double slip;
    double lon, lat, depth;
    double strike;
    double dip;
    double rake;
    double length, width;
    double mu;

    cOkadaFault();
    ~cOkadaFault();
    int    read(char *faultparam);
    int    check();
    double getMw();
    double getM0();
    double getZtop();
    int    global2local(double glon, double glat, double &lx, double &ly);
    int    local2global(double lx, double ly, double &glon, double &glat);
    int    getDeformArea(double &lonmin, double &lonmax, double &latmin, double &latmax);
    int    calculate(double lon, double lat, double &uz, double &ulon, double &ulat);
    int    calculate(double lon, double lat, double &uz);
};

int okada(double L, double W, double D, double sinD, double cosD, double U1, double U2, double x, double y, int flag_xy, double *Ux, double *Uy, double *Uz);

#endif // OKADAFAULT_H
