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


#ifndef EWDEFS_H
#define EWDEFS_H

#define MAX_VARS_PER_NODE 12

#define iD    0
#define iH    1
#define iHmax 2
#define iM    3
#define iN    4
#define iR1   5
#define iR2   6
#define iR3   7
#define iR4   8
#define iR5   9
#define iTime 10
#define iTopo 11

// Global data
struct EWPARAMS {
    char *modelName;
    char *modelSubset;
    char *fileBathymetry;
    char *fileSource;
    char *filePOIs;
    int   dt;
    int   time;
    int   timeMax;
    int   poiDt;
    int   poiReport;
    int   outDump;
    int   outProgress;
    int   outPropagation;
    int   coriolis;
    float dmin;
    float poiDistMax;
    float poiDepthMin;
    float poiDepthMax;
    float ssh0ThresholdRel;
    float ssh0ThresholdAbs;
    float sshClipThreshold;
    float sshZeroThreshold;
    float sshTransparencyThreshold;
    float sshArrivalThreshold;
    bool  gpu;
    bool  adjustZtop;
    bool  verbose;
};

int ewStep();
int ewStepCor();

extern struct EWPARAMS Par;
extern int             NLon, NLat;
extern double          LonMin, LonMax, LatMin, LatMax;
extern double          DLon, DLat; // steps in grad
extern double          Dx, Dy;     // steps in m, dx must be multiplied by cos(y) before use
extern float          *R6;
extern float          *C1;
extern float          *C2;
extern float          *C3;
extern float          *C4;
extern int             Imin;
extern int             Imax;
extern int             Jmin;
extern int             Jmax;

#define idx(j, i) ((i - 1) * NLat + j - 1)
#define getLon(i) (LonMin + (i - 1) * DLon)
#define getLat(j) (LatMin + (j - 1) * DLat)

#endif /* EWDEFS_H */
