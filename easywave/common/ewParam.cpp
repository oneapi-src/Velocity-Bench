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

struct EWPARAMS Par;

int ewParam(int argc, char **argv)
// Process command line arguments and/or use default
{
    int argn, ierr;

    /* TODO: optimize argument handling */

    // Obligatory command line parameters

    // Bathymetry
    if ((argn = utlCheckCommandLineOption(argc, argv, "grid", 4)) != 0) {
        /* TODO: strdup not necessary here because all arguments in argv reside until program exit -> memory leak */
        Par.fileBathymetry = strdup(argv[argn + 1]);
    } else
        return -1;

    // Source: Okada faults or Surfer grid
    if ((argn = utlCheckCommandLineOption(argc, argv, "source", 6)) != 0) {
        Par.fileSource = strdup(argv[argn + 1]);
    } else
        return -1;

    // Simulation time, [sec]
    if ((argn = utlCheckCommandLineOption(argc, argv, "time", 4)) != 0) {
        Par.timeMax = atoi(argv[argn + 1]);
        Par.timeMax *= 60;
    } else
        return -1;

    // Optional parameters or their default values

    // Model name
    if ((argn = utlCheckCommandLineOption(argc, argv, "label", 3)) != 0) {
        Par.modelName = strdup(argv[argn + 1]);
    } else
        Par.modelName = strdup("eWave");

    // Deactivate logging
    if ((argn = utlCheckCommandLineOption(argc, argv, "nolog", 5)) != 0)
        ;
    else {
        Log.start("easywave.log");
        Log.timestamp_disable();
    }

    // Use Coriolis force
    if ((argn = utlCheckCommandLineOption(argc, argv, "coriolis", 3)) != 0)
        Par.coriolis = 1;
    else
        Par.coriolis = 0;

    // Periodic dumping of mariograms and cumulative 2D-plots (wavemax, arrival times), [sec]
    if ((argn = utlCheckCommandLineOption(argc, argv, "dump", 4)) != 0)
        Par.outDump = atoi(argv[argn + 1]);
    else
        Par.outDump = 0;

    // Reporting simulation progress, [sec model time]
    if ((argn = utlCheckCommandLineOption(argc, argv, "progress", 4)) != 0)
        Par.outProgress = (int)(atof(argv[argn + 1]) * 60);
    else
        Par.outProgress = 600;

    // 2D-wave propagation output, [sec model time]
    if ((argn = utlCheckCommandLineOption(argc, argv, "propagation", 4)) != 0)
        Par.outPropagation = (int)(atof(argv[argn + 1]) * 60);
    else
        Par.outPropagation = 300;

    // minimal calculation depth, [m]
    if ((argn = utlCheckCommandLineOption(argc, argv, "min_depth", 9)) != 0)
        Par.dmin = (float)atof(argv[argn + 1]);
    else
        Par.dmin = 10.;

    // timestep, [sec]
    if ((argn = utlCheckCommandLineOption(argc, argv, "step", 4)) != 0)
        Par.dt = atoi(argv[argn + 1]);
    else
        Par.dt = 0; // will be estimated automatically

    // Initial uplift: relative threshold
    if ((argn = utlCheckCommandLineOption(argc, argv, "ssh0_rel", 8)) != 0)
        Par.ssh0ThresholdRel = (float)atof(argv[argn + 1]);
    else
        Par.ssh0ThresholdRel = 0.01;

    // Initial uplift: absolute threshold, [m]
    if ((argn = utlCheckCommandLineOption(argc, argv, "ssh0_abs", 8)) != 0)
        Par.ssh0ThresholdAbs = (float)atof(argv[argn + 1]);
    else
        Par.ssh0ThresholdAbs = 0.0;

    // Threshold for 2-D arrival time (0 - do not calculate), [m]
    if ((argn = utlCheckCommandLineOption(argc, argv, "ssh_arrival", 9)) != 0)
        Par.sshArrivalThreshold = (float)atof(argv[argn + 1]);
    else
        Par.sshArrivalThreshold = 0.001;

    // Threshold for clipping of expanding computational area, [m]
    if ((argn = utlCheckCommandLineOption(argc, argv, "ssh_clip", 8)) != 0)
        Par.sshClipThreshold = (float)atof(argv[argn + 1]);
    else
        Par.sshClipThreshold = 1.e-4;

    // Threshold for resetting the small ssh (keep expanding area from unnesessary growing), [m]
    if ((argn = utlCheckCommandLineOption(argc, argv, "ssh_zero", 8)) != 0)
        Par.sshZeroThreshold = (float)atof(argv[argn + 1]);
    else
        Par.sshZeroThreshold = 1.e-5;

    // Threshold for transparency (for png-output), [m]
    if ((argn = utlCheckCommandLineOption(argc, argv, "ssh_transparency", 8)) != 0)
        Par.sshTransparencyThreshold = (float)atof(argv[argn + 1]);
    else
        Par.sshTransparencyThreshold = 0.0;

    // Points Of Interest (POIs) input file
    if ((argn = utlCheckCommandLineOption(argc, argv, "poi", 3)) != 0) {
        Par.filePOIs = strdup(argv[argn + 1]);
    } else
        Par.filePOIs = NULL;

    // POI fitting: max search distance, [km]
    if ((argn = utlCheckCommandLineOption(argc, argv, "poi_search_dist", 15)) != 0)
        Par.poiDistMax = (float)atof(argv[argn + 1]);
    else
        Par.poiDistMax = 10.0;
    Par.poiDistMax *= 1000.;

    // POI fitting: min depth, [m]
    if ((argn = utlCheckCommandLineOption(argc, argv, "poi_min_depth", 13)) != 0)
        Par.poiDepthMin = (float)atof(argv[argn + 1]);
    else
        Par.poiDepthMin = 1.0;

    // POI fitting: max depth, [m]
    if ((argn = utlCheckCommandLineOption(argc, argv, "poi_max_depth", 13)) != 0)
        Par.poiDepthMax = (float)atof(argv[argn + 1]);
    else
        Par.poiDepthMax = 10000.0;

    // report of POI loading
    if ((argn = utlCheckCommandLineOption(argc, argv, "poi_report", 7)) != 0)
        Par.poiReport = 1;
    else
        Par.poiReport = 0;

    // POI output interval, [sec]
    if ((argn = utlCheckCommandLineOption(argc, argv, "poi_dt_out", 10)) != 0)
        Par.poiDt = atoi(argv[argn + 1]);
    else
        Par.poiDt = 30;

    if ((argn = utlCheckCommandLineOption(argc, argv, "gpu", 3)) != 0)
        Par.gpu = true;
    else
        Par.gpu = false;

    if ((argn = utlCheckCommandLineOption(argc, argv, "adjust_ztop", 11)) != 0)
        Par.adjustZtop = true;
    else
        Par.adjustZtop = false;

    if ((argn = utlCheckCommandLineOption(argc, argv, "verbose", 7)) != 0)
        Par.verbose = true;
    else
        Par.verbose = false;

    return 0;
}

void ewLogParams(void)
{
    Log.print("\nModel parameters for this simulation:");
    Log.print("timestep: %d sec", Par.dt);
    Log.print("max time: %g min", (float)Par.timeMax / 60);
    Log.print("poi_dt_out: %d sec", Par.poiDt);
    Log.print("poi_report: %s", (Par.poiReport ? "yes" : "no"));
    Log.print("poi_search_dist: %g km", Par.poiDistMax / 1000.);
    Log.print("poi_min_depth: %g m", Par.poiDepthMin);
    Log.print("poi_max_depth: %g m", Par.poiDepthMax);
    Log.print("coriolis: %s", (Par.coriolis ? "yes" : "no"));
    Log.print("min_depth: %g m", Par.dmin);
    Log.print("ssh0_rel: %g", Par.ssh0ThresholdRel);
    Log.print("ssh0_abs: %g m", Par.ssh0ThresholdAbs);
    Log.print("ssh_arrival: %g m", Par.sshArrivalThreshold);
    Log.print("ssh_clip: %g m", Par.sshClipThreshold);
    Log.print("ssh_zero: %g m", Par.sshZeroThreshold);
    Log.print("ssh_transparency: %g m\n", Par.sshTransparencyThreshold);

    return;
}
