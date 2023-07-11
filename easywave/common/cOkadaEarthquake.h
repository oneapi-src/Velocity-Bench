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

#ifndef OKADAEARTHQUAKE_H
#define OKADAEARTHQUAKE_H

#include "cOgrd.h"
#include "cSphere.h"
#include "cOkadaFault.h"

class cOkadaEarthquake
{
  protected:
    int finalized;
    int getDeformArea(int round, double &lonmin, double &lonmax, double &latmin, double &latmax);
    int setGrid(cOgrd &u);

    cOkadaEarthquake(cOkadaEarthquake const &RHS)            = delete;
    cOkadaEarthquake &operator=(cOkadaEarthquake const &RHS) = delete;

  public:
    int          nfault; // total nuber of Okada faults
    double       m0;     // total earthquake moment
    cOkadaFault *fault;  // array of composing faults

    cOkadaEarthquake();
    ~cOkadaEarthquake();
    int    read(char *fname);
    int    finalizeInput();
    double getM0();
    double getMw();
    int    calculate(double lon, double lat, double &uz);
    int    calculate(double lon, double lat, double &uz, double &ulon, double &ulat);
    int    calculate(cObsArray &arr);
    int    calculate(cOgrd &uZ);
    int    calculate(cOgrd &uZ, cOgrd &uLon, cOgrd &uLat);
    char  *sprint();
};

#endif // OKADAEARTHQUAKE_H
