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

#ifndef EW_NODE_H
#define EW_NODE_H

#include <stdlib.h>
#include <string.h>

#include "ewdefs.h"

#define CHKRET(x)    \
    if ((x) == NULL) \
        return 1;

typedef float Float[MAX_VARS_PER_NODE];

class CNode
{

  public:
    virtual ~CNode(){};
    virtual float &operator()(const int idx1, const int idx2) = 0;
    virtual int    mallocMem()                                = 0;
    virtual int    copyToGPU()                                = 0;
    virtual int    copyFromGPU()                              = 0;
    virtual int    copyIntermediate()                         = 0;
    virtual int    copyPOIs()                                 = 0;
    virtual int    freeMem()                                  = 0;
    virtual int    run()                                      = 0;

    virtual void initMemory(int index, int val) = 0;
};

class CStructNode : public CNode
{

  public:
    Float *node;

  public:
    inline float &operator()(const int idx1, const int idx2)
    {

        return node[idx1][idx2];
    }

    void initMemory(int index, int val)
    {

        int m;
        for (int i = 1; i <= NLon; i++) {
            for (int j = 1; j <= NLat; j++) {
                m                          = idx(j, i);
                this->operator()(m, index) = val;
            }
        }
    }

    int mallocMem()
    {

        CHKRET(this->node = (Float *)malloc(sizeof(Float) * NLon * NLat));

        /* FIXME: remove global variables */
        CHKRET(R6 = (float *)malloc(sizeof(float) * (NLat + 1)));
        CHKRET(C1 = (float *)malloc(sizeof(float) * (NLon + 1)));
        CHKRET(C3 = (float *)malloc(sizeof(float) * (NLon + 1)));
        CHKRET(C2 = (float *)malloc(sizeof(float) * (NLat + 1)));
        CHKRET(C4 = (float *)malloc(sizeof(float) * (NLat + 1)));

        return 0;
    }

    int freeMem()
    {

        free(this->node);
        free(R6);
        free(C1);
        free(C2);
        free(C3);
        free(C4);

        return 0;
    }

    int run()
    {

        if (Par.coriolis)
            return ewStepCor();

        return ewStep();
    }

    int copyToGPU() { return 0; }
    int copyFromGPU() { return 0; }
    int copyIntermediate() { return 0; }
    int copyPOIs() { return 0; }
};

#pragma pack(push, 1)
class CArrayNode : public CNode
{

  protected:
    float *d;
    float *h;
    float *hMax;
    float *fM;
    float *fN;
    float *cR1;
    float *cR2;
    float *cR3;
    float *cR4;
    float *cR5;
    float *tArr;
    float *topo;

    float *d_1D_aligned;
    float *h_1D_aligned;
    float *hMax_1D_aligned;
    float *fM_1D_aligned;
    float *fN_1D_aligned;
    float *cR1_1D_aligned;
    float *cR2_1D_aligned;
    float *cR4_1D_aligned;
    float *tArr_1D_aligned;

  public:
    virtual float &operator()(const int idx1, const int idx2)
    {

        return ((float **)&d)[idx2][idx1];
    }

    void *getBuf(int idx) { return ((float **)&d)[idx]; }

    virtual void initMemory(int index, int /* val */)
    {

        memset(getBuf(index), 0, NLat * NLon * sizeof(float));
    }

    virtual int mallocMem()
    {

        CHKRET(this->d = (float *)malloc(sizeof(float) * NLon * NLat));
        CHKRET(this->h = (float *)malloc(sizeof(float) * NLon * NLat));
        CHKRET(this->hMax = (float *)malloc(sizeof(float) * NLon * NLat));
        CHKRET(this->fM = (float *)malloc(sizeof(float) * NLon * NLat));
        CHKRET(this->fN = (float *)malloc(sizeof(float) * NLon * NLat));
        CHKRET(this->cR1 = (float *)malloc(sizeof(float) * NLon * NLat));
        CHKRET(this->cR2 = (float *)malloc(sizeof(float) * NLon * NLat));
        CHKRET(this->cR3 = (float *)malloc(sizeof(float) * NLon * NLat));
        CHKRET(this->cR4 = (float *)malloc(sizeof(float) * NLon * NLat));
        CHKRET(this->cR5 = (float *)malloc(sizeof(float) * NLon * NLat));
        CHKRET(this->tArr = (float *)malloc(sizeof(float) * NLon * NLat));
        CHKRET(this->topo = (float *)malloc(sizeof(float) * NLon * NLat));

        /* FIXME: remove global variables */
        CHKRET(R6 = (float *)malloc(sizeof(float) * (NLat + 1)));
        CHKRET(C1 = (float *)malloc(sizeof(float) * (NLon + 1)));
        CHKRET(C3 = (float *)malloc(sizeof(float) * (NLon + 1)));
        CHKRET(C2 = (float *)malloc(sizeof(float) * (NLat + 1)));
        CHKRET(C4 = (float *)malloc(sizeof(float) * (NLat + 1)));

        return 0;
    }

    virtual int freeMem()
    {

        free(this->d);
        free(this->h);
        free(this->hMax);
        free(this->fM);
        free(this->fN);
        free(this->cR1);
        free(this->cR2);
        free(this->cR3);
        free(this->cR4);
        free(this->cR5);
        free(this->tArr);
        free(this->topo);

        free(R6);
        free(C1);
        free(C2);
        free(C3);
        free(C4);

        return 0;
    }

    virtual int run()
    {

        if (Par.coriolis)
            return ewStepCor();

        return ewStep();
    }

    virtual int copyToGPU() { return 0; }
    virtual int copyFromGPU() { return 0; }
    virtual int copyIntermediate() { return 0; }
    virtual int copyPOIs() { return 0; }
};
#pragma pack(pop)

#endif /* EW_NODE_H */
