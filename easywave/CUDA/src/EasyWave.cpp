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


#define HEADER "easyWave ver.2013-04-11"

#include "Utilities.h"
#include "FileHandler.h"

#define CPP_MODULE "MAIN"
#include "Logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cassert>
#include "utilits.h"
#include "easywave.h"

//#ifdef __CUDACC__
#include "ewGpuNode.cuh"
//#endif

CNode *gNode;

double diff(timespec start, timespec end)
{

    timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec  = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        temp.tv_sec  = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }

    return (double)((double)temp.tv_nsec / 1000000000.0 + (double)temp.tv_sec);
}

int commandLineHelp(void);

int main(int argc, char **argv)
{
    Timer tWallClock("WallClock");
    tWallClock.Start();
    LOG("Starting CUDA main program. Process ID: " << Utility::GetProcessID());
    Utility::QueryCUDADevice();

    std::vector<std::string> eWaveFiles(Utility::FileHandler::GetFilesFromDirectory(Utility::FileHandler::GetCurrentDirectory()));
    if (!eWaveFiles.empty()) {
        LOG("Attempting to clean up previous eWave tsunami files in " << Utility::FileHandler::GetCurrentDirectory());
        eWaveFiles.erase(std::remove_if(eWaveFiles.begin(), eWaveFiles.end(), [](std::string const &sPath) { return sPath.rfind("eWave") == std::string::npos; }), eWaveFiles.end());
        size_t const uNumberOfeWaveFiles(eWaveFiles.size());
        size_t const uNumberOfDeletedFiles(Utility::FileHandler::RemoveFiles(eWaveFiles));
        if (uNumberOfDeletedFiles != uNumberOfeWaveFiles)
            LOG_WARNING("Only deleted " << uNumberOfDeletedFiles << " out of " << uNumberOfeWaveFiles << " eWave files found");
        LOG("Clean up completed");
    }

    int      ierr, argn;
    long int elapsed;
    int      lastProgress, lastPropagation, lastDump;
    int      loop;

    double dAccumulateIOReadTime(0.0);
    ////printf(HEADER);
    Err.setchannel(MSG_OUTFILE);

    // Read parameters from command line and use default
    ierr = ewParam(argc, argv);
    if (ierr)
        return commandLineHelp();

    // Log command line
    /* FIXME: buffer overflow */
    std::stringstream ss;
    for (argn = 1; argn < argc; argn++) {
        ss << " ";
        ss << argv[argn];
    }
    Log.print("%s", ss.str().c_str());

    gNode = new CGpuNode();
    assert(gNode != nullptr);
    CNode &Node = *gNode;

    // Read bathymetry
    ierr = ewLoadBathymetry(dAccumulateIOReadTime);
    if (ierr)
        return ierr;

    // Read points of interest
    ierr = ewLoadPOIs();
    if (ierr)
        return ierr;

    // Init tsunami with faults or uplift-grid
    ierr = ewSource(dAccumulateIOReadTime); // I/O (FileHandler / fread)
    if (ierr)
        return ierr;
    Log.print("Read source from %s", Par.fileSource);

    // Write model parameters into the log
    ewLogParams();

    if (Par.outPropagation) {
        ewStart2DOutput();
    }

    Node.copyToGPU();

    // Main loop
    Log.print("Starting main loop...");
#ifdef SHOW_GRID
    printf("Jmin:Jmax=%d:%d Imin:Imax=%d:%d (Jmax-Jmin+1)*(Imax-Imin+1)=(%d*%d)=%d\n", Jmin, Jmax, Imin, Imax, (Jmax - Jmin + 1), (Imax - Imin + 1), (Jmax - Jmin + 1) * (Imax - Imin + 1));
    Log.print("Jmin:Jmax=%d:%d Imin:Imax=%d:%d (Jmax-Jmin+1)*(Imax-Imin+1)=(%d*%d)=%d\n", Jmin, Jmax, Imin, Imax, (Jmax - Jmin + 1), (Imax - Imin + 1), (Jmax - Jmin + 1) * (Imax - Imin + 1));
#endif

    timespec start, inter; // Used for progress output of ssh files  //, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    double                                dAccumulatedIOWriteTime(0.0);
    std::chrono::steady_clock::time_point tpStart;

    for (Par.time = 0, loop = 1, lastProgress = Par.outProgress, lastPropagation = Par.outPropagation, lastDump = 0;
         Par.time <= Par.timeMax;
         loop++, Par.time += Par.dt, lastProgress += Par.dt, lastPropagation += Par.dt) {

        /* FIXME: check if Par.poiDt can be used for those purposes */
        if (Par.filePOIs && Par.poiDt && ((Par.time / Par.poiDt) * Par.poiDt == Par.time)) { // Is this needed?
            Node.copyPOIs();                                                                 // Copies the POIs
            ewSavePOIs();                                                                    // Saves to a file
        }

        Node.run();

        clock_gettime(CLOCK_MONOTONIC, &inter);
        elapsed = diff(start, inter) * 1000;

        if (Par.outProgress) { // Time is now shown using steady_clock. TimeSpec timer types will be used for dumping .ssh files (See if (Par.outdump) block)
            if (lastProgress >= Par.outProgress) {
                LOG("Model time: " << utlTimeSplitString(Par.time)); //// << " Elapsed: " << tCumulatedComputeTime.GetTimeAsString(Timer::Units::SECONDS));
                /////printf( "Model time = %s,   elapsed: %ld msec\n", utlTimeSplitString(Par.time), elapsed );
                Log.print("Model time = %s,   elapsed: %ld msec", utlTimeSplitString(Par.time), elapsed);

#ifdef SHOW_GRID
                printf("Jmin:Jmax=%d:%d Imin:Imax=%d:%d (Jmax-Jmin+1)*(Imax-Imin+1)=(%d*%d)=%d\n", Jmin, Jmax, Imin, Imax, (Jmax - Jmin + 1), (Imax - Imin + 1), (Jmax - Jmin + 1) * (Imax - Imin + 1));
                Log.print("Jmin:Jmax=%d:%d Imin:Imax=%d:%d (Jmax-Jmin+1)*(Imax-Imin+1)=(%d*%d)=%d\n", Jmin, Jmax, Imin, Imax, (Jmax - Jmin + 1), (Imax - Imin + 1), (Jmax - Jmin + 1) * (Imax - Imin + 1));
#endif

                lastProgress = 0;
            }
        }

        fflush(stdout);

        if (Par.outPropagation) { // Outputs 2D-wave propagation
            if (lastPropagation >= Par.outPropagation) {
                Node.copyIntermediate();
                tpStart = std::chrono::steady_clock::now();
                ewOut2D();
                dAccumulatedIOWriteTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - tpStart).count();
                lastPropagation = 0;
            }
        }

        if (Par.outDump) {
            if ((elapsed - lastDump) >= Par.outDump) {
                Node.copyIntermediate();
                ewDumpPOIs();
                ewDump2D();
                lastDump = elapsed;
            }
        }

    } // main loop
    LOG("Compute loop completed");

    // clock_gettime(CLOCK_MONOTONIC, &end);
    Log.print("Finishing main loop");

    /* TODO: check if theses calls can be combined */
    Node.copyIntermediate();
    Node.copyFromGPU();

    // Final output
    Log.print("Final dump...");
    ewDumpPOIs();
    ewDump2D();

    Node.freeMem();
    reinterpret_cast<CGpuNode *>(gNode)->PrintTimingStats();

    delete gNode;

    LOG("Program successfully completed");
    tWallClock.Stop();
    LOG("I/O Time            : " << dAccumulatedIOWriteTime - dAccumulateIOReadTime << " s");
    LOG("Total Execution Time: " << tWallClock.GetTime() - dAccumulatedIOWriteTime - dAccumulateIOReadTime << " s");
    return 0;
}

//========================================================================
int commandLineHelp(void)
{
    printf("Usage: easywave  -grid ...  -source ...  -time ... [optional parameters]\n");
    printf("-grid ...         bathymetry in GoldenSoftware(C) GRD format (text or binary)\n");
    printf("-source ...       input wave either als GRD-file or file with Okada faults\n");
    printf("-time ...         simulation time in [min]\n");
    printf("Optional parameters:\n");
    printf("-step ...         simulation time step, default- estimated from bathymetry\n");
    printf("-coriolis         use Coriolis fource, default- no\n");
    printf("-poi ...          POIs file\n");
    printf("-label ...        model name, default- 'eWave'\n");
    printf("-progress ...     show simulation progress each ... minutes, default- 10\n");
    printf("-propagation ...  write wave propagation grid each ... minutes, default- 5\n");
    printf("-dump ...         make solution dump each ... physical seconds, default- 0\n");
    printf("-nolog            deactivate logging\n");
    printf("-poi_dt_out ...   output time step for mariograms in [sec], default- 30\n");
    printf("-poi_search_dist ...  in [km], default- 10\n");
    printf("-poi_min_depth ...    in [m], default- 1\n");
    printf("-poi_max_depth ...    in [m], default- 10 000\n");
    printf("-poi_report       enable POIs loading report, default- disabled\n");
    printf("-ssh0_rel ...     relative threshold for initial wave, default- 0.01\n");
    printf("-ssh0_abs ...     absolute threshold for initial wave in [m], default- 0\n");
    printf("-ssh_arrival ...  threshold for arrival times in [m], default- 0.001\n");
    printf("                  negative value considered as relative threshold\n");
    printf("-gpu              start GPU version of EasyWave (requires a CUDA capable device)\n");
    printf("-verbose          generate verbose output on stdout\n");
    printf("\nExample:\n");
    printf("\t easyWave -grid gebcoIndonesia.grd  -source fault.inp  -time 120\n\n");

    return -1;
}
