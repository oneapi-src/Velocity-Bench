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

// Time stepping
#include <stdio.h>
#include <stdlib.h>

#include "utilits.h"
#include "easywave.h"
#include <cmath>

/* TODO: still not perfect */
#define Node(idx1, idx2) Node.node[idx1][idx2]
#define CNode            CStructNode
#define gNode            ((CStructNode *)gNode)

int ewStep(void)
{
    int   i, j, enlarge;
    float absH, v1, v2;
    int   m;

    CNode &Node = *gNode;

// sea floor topography (mass conservation)
#pragma omp parallel for default(shared) private(i, j, m, absH)
    for (i = Imin; i <= Imax; i++) {
        for (j = Jmin; j <= Jmax; j++) {

            m = idx(j, i);

            if (Node(m, iD) == 0)
                continue;

            Node(m, iH) = Node(m, iH) - Node(m, iR1) * (Node(m, iM) - Node(m - NLat, iM) + Node(m, iN) * R6[j] - Node(m - 1, iN) * R6[j - 1]);

            absH = fabs(Node(m, iH));

            if (absH < Par.sshZeroThreshold)
                Node(m, iH) = 0.;

            if (Node(m, iH) > Node(m, iHmax))
                Node(m, iHmax) = Node(m, iH);

            if (Par.sshArrivalThreshold && Node(m, iTime) < 0 && absH > Par.sshArrivalThreshold)
                Node(m, iTime) = (float)Par.time;
        }
    }

    // open bondary conditions
    if (Jmin <= 2) {
        for (i = 2; i <= (NLon - 1); i++) {
            m           = idx(1, i);
            Node(m, iH) = sqrt(pow(Node(m, iN), 2.) + 0.25 * pow((Node(m, iM) + Node(m - NLat, iM)), 2.)) *
                          C1[i];
            if (Node(m, iN) > 0)
                Node(m, iH) = -Node(m, iH);
        }
    }
    if (Imin <= 2) {
        for (j = 2; j <= (NLat - 1); j++) {
            m           = idx(j, 1);
            Node(m, iH) = sqrt(pow(Node(m, iM), 2.) + 0.25 * pow((Node(m, iN) + Node(m - 1, iN)), 2.)) *
                          C2[j];
            if (Node(m, iM) > 0)
                Node(m, iH) = -Node(m, iH);
        }
    }
    if (Jmax >= (NLat - 1)) {
        for (i = 2; i <= (NLon - 1); i++) {
            m           = idx(NLat, i);
            Node(m, iH) = sqrt(pow(Node(m - 1, iN), 2.) + 0.25 * pow((Node(m, iM) + Node(m - 1, iM)), 2.)) *
                          C3[i];
            if (Node(m - 1, iN) < 0)
                Node(m, iH) = -Node(m, iH);
        }
    }
    if (Imax >= (NLon - 1)) {
        for (j = 2; j <= (NLat - 1); j++) {
            m           = idx(j, NLon);
            Node(m, iH) = sqrt(pow(Node(m - NLat, iM), 2.) + 0.25 * pow((Node(m, iN) + Node(m - 1, iN)), 2.)) *
                          C4[j];
            if (Node(m - NLat, iM) < 0)
                Node(m, iH) = -Node(m, iH);
        }
    }
    if (Jmin <= 2) {
        m           = idx(1, 1);
        Node(m, iH) = sqrt(pow(Node(m, iM), 2.) + pow(Node(m, iN), 2.)) * C1[1];
        if (Node(m, iN) > 0)
            Node(m, iH) = -Node(m, iH);
        m = idx(1, NLon);
        Node(m, iH) =
            sqrt(pow(Node(m - NLat, iM), 2.) + pow(Node(m, iN), 2.)) * C1[NLon];
        if (Node(m, iN) > 0)
            Node(m, iH) = -Node(m, iH);
    }
    if (Jmin >= (NLat - 1)) {
        m           = idx(NLat, 1);
        Node(m, iH) = sqrt(pow(Node(m, iM), 2.) + pow(Node(m - 1, iN), 2.)) * C3[1];
        if (Node(m - 1, iN) < 0)
            Node(m, iH) = -Node(m, iH);
        m = idx(NLat, NLon);
        Node(m, iH) =
            sqrt(pow(Node(m - NLat, iM), 2.) + pow(Node(m - 1, iN), 2.)) * C3[NLon];
        if (Node(m - 1, iN) < 0)
            Node(m, iH) = -Node(m, iH);
    }

// moment conservation
#pragma omp parallel for default(shared) private(i, j, m)
    for (i = Imin; i <= Imax; i++) {
        for (j = Jmin; j <= Jmax; j++) {

            m = idx(j, i);

            if ((Node(m, iD) * Node(m + NLat, iD)) != 0)
                Node(m, iM) = Node(m, iM) - Node(m, iR2) * (Node(m + NLat, iH) - Node(m, iH));

            if ((Node(m, iD) * Node(m + 1, iD)) != 0)
                Node(m, iN) = Node(m, iN) - Node(m, iR4) * (Node(m + 1, iH) - Node(m, iH));
        }
    }
    // open boundaries
    if (Jmin <= 2) {
        for (i = 1; i <= (NLon - 1); i++) {
            m           = idx(1, i);
            Node(m, iM) = Node(m, iM) - Node(m, iR2) * (Node(m + NLat, iH) - Node(m, iH));
        }
    }
    if (Imin <= 2) {
        for (j = 1; j <= NLat; j++) {
            m           = idx(j, 1);
            Node(m, iM) = Node(m, iM) - Node(m, iR2) * (Node(m + NLat, iH) - Node(m, iH));
        }
    }
    if (Jmax >= (NLat - 1)) {
        for (i = 1; i <= (NLon - 1); i++) {
            m           = idx(NLat, i);
            Node(m, iM) = Node(m, iM) - Node(m, iR2) * (Node(m + NLat, iH) - Node(m, iH));
        }
    }
    if (Imin <= 2) {
        for (j = 1; j <= (NLat - 1); j++) {
            m           = idx(j, 1);
            Node(m, iN) = Node(m, iN) - Node(m, iR4) * (Node(m + 1, iH) - Node(m, iH));
        }
    }
    if (Jmin <= 2) {
        for (i = 1; i <= NLon; i++) {
            m           = idx(1, i);
            Node(m, iN) = Node(m, iN) - Node(m, iR4) * (Node(m + 1, iH) - Node(m, iH));
        }
    }
    if (Imax >= (NLon - 1)) {
        for (j = 1; j <= (NLat - 1); j++) {
            m           = idx(j, NLon);
            Node(m, iN) = Node(m, iN) - Node(m, iR4) * (Node(m + 1, iH) - Node(m, iH));
        }
    }

    // calculation area for the next step
    if (Imin > 2) {
        for (enlarge = 0, j = Jmin; j <= Jmax; j++) {
            if (fabs(Node(idx(j, Imin + 2), iH)) > Par.sshClipThreshold) {
                enlarge = 1;
                break;
            }
        }
        if (enlarge) {
            Imin--;
            if (Imin < 2)
                Imin = 2;
        }
    }
    if (Imax < (NLon - 1)) {
        for (enlarge = 0, j = Jmin; j <= Jmax; j++) {
            if (fabs(Node(idx(j, Imax - 2), iH)) > Par.sshClipThreshold) {
                enlarge = 1;
                break;
            }
        }
        if (enlarge) {
            Imax++;
            if (Imax > (NLon - 1))
                Imax = NLon - 1;
        }
    }
    if (Jmin > 2) {
        for (enlarge = 0, i = Imin; i <= Imax; i++) {
            if (fabs(Node(idx(Jmin + 2, i), iH)) > Par.sshClipThreshold) {
                enlarge = 1;
                break;
            }
        }
        if (enlarge) {
            Jmin--;
            if (Jmin < 2)
                Jmin = 2;
        }
    }
    if (Jmax < (NLat - 1)) {
        for (enlarge = 0, i = Imin; i <= Imax; i++) {
            if (fabs(Node(idx(Jmax - 2, i), iH)) > Par.sshClipThreshold) {
                enlarge = 1;
                break;
            }
        }
        if (enlarge) {
            Jmax++;
            if (Jmax > (NLat - 1))
                Jmax = NLat - 1;
        }
    }

    return 0;
}

int ewStepCor(void)
{
    int   i, j, enlarge;
    float absH, v1, v2;
    int   m;

    CNode &Node = *gNode;

// sea floor topography (mass conservation)
#pragma omp parallel for default(shared) private(i, j, m, absH)
    for (i = Imin; i <= Imax; i++) {
        for (j = Jmin; j <= Jmax; j++) {

            m = idx(j, i);

            if (Node(m, iD) == 0)
                continue;

            Node(m, iH) = Node(m, iH) - Node(m, iR1) * (Node(m, iM) - Node(m - NLat, iM) + Node(m, iN) * R6[j] - Node(m - 1, iN) * R6[j - 1]);

            absH = fabs(Node(m, iH));

            if (absH < Par.sshZeroThreshold)
                Node(m, iH) = 0.;

            if (Node(m, iH) > Node(m, iHmax))
                Node(m, iHmax) = Node(m, iH);

            if (Par.sshArrivalThreshold && Node(m, iTime) < 0 && absH > Par.sshArrivalThreshold)
                Node(m, iTime) = (float)Par.time;
        }
    }

    // open bondary conditions
    if (Jmin <= 2) {
        for (i = 2; i <= (NLon - 1); i++) {
            m           = idx(1, i);
            Node(m, iH) = sqrt(pow(Node(m, iN), 2.) + 0.25 * pow((Node(m, iM) + Node(m - NLat, iM)), 2.)) *
                          C1[i];
            if (Node(m, iN) > 0)
                Node(m, iH) = -Node(m, iH);
        }
    }
    if (Imin <= 2) {
        for (j = 2; j <= (NLat - 1); j++) {
            m           = idx(j, 1);
            Node(m, iH) = sqrt(pow(Node(m, iM), 2.) + 0.25 * pow((Node(m, iN) + Node(m - 1, iN)), 2.)) *
                          C2[j];
            if (Node(m, iM) > 0)
                Node(m, iH) = -Node(m, iH);
        }
    }
    if (Jmax >= (NLat - 1)) {
        for (i = 2; i <= (NLon - 1); i++) {
            m           = idx(NLat, i);
            Node(m, iH) = sqrt(pow(Node(m - 1, iN), 2.) + 0.25 * pow((Node(m, iM) + Node(m - 1, iM)), 2.)) *
                          C3[i];
            if (Node(m - 1, iN) < 0)
                Node(m, iH) = -Node(m, iH);
        }
    }
    if (Imax >= (NLon - 1)) {
        for (j = 2; j <= (NLat - 1); j++) {
            m           = idx(j, NLon);
            Node(m, iH) = sqrt(pow(Node(m - NLat, iM), 2.) + 0.25 * pow((Node(m, iN) + Node(m - 1, iN)), 2.)) *
                          C4[j];
            if (Node(m - NLat, iM) < 0)
                Node(m, iH) = -Node(m, iH);
        }
    }
    if (Jmin <= 2) {
        m           = idx(1, 1);
        Node(m, iH) = sqrt(pow(Node(m, iM), 2.) + pow(Node(m, iN), 2.)) * C1[1];
        if (Node(m, iN) > 0)
            Node(m, iH) = -Node(m, iH);
        m = idx(1, NLon);
        Node(m, iH) =
            sqrt(pow(Node(m - NLat, iM), 2.) + pow(Node(m, iN), 2.)) * C1[NLon];
        if (Node(m, iN) > 0)
            Node(m, iH) = -Node(m, iH);
    }
    if (Jmin >= (NLat - 1)) {
        m           = idx(NLat, 1);
        Node(m, iH) = sqrt(pow(Node(m, iM), 2.) + pow(Node(m - 1, iN), 2.)) * C3[1];
        if (Node(m - 1, iN) < 0)
            Node(m, iH) = -Node(m, iH);
        m = idx(NLat, NLon);
        Node(m, iH) =
            sqrt(pow(Node(m - NLat, iM), 2.) + pow(Node(m - 1, iN), 2.)) * C3[NLon];
        if (Node(m - 1, iN) < 0)
            Node(m, iH) = -Node(m, iH);
    }

// moment conservation
// longitudial flux update
#pragma omp parallel for default(shared) private(i, j, m, v1, v2)
    for (i = Imin; i <= Imax; i++) {
        for (j = Jmin; j <= Jmax; j++) {

            m = idx(j, i);

            if ((Node(m, iD) * Node(m + NLat, iD)) == 0)
                continue;

            v1          = Node(m + NLat, iH) - Node(m, iH);
            v2          = Node(m - 1, iN) + Node(m, iN) + Node(m + NLat, iN) + Node(m + NLat - 1, iN);
            Node(m, iM) = Node(m, iM) - Node(m, iR2) * v1 + Node(m, iR3) * v2;
        }
    }
    // open boundaries
    if (Jmin <= 2) {
        for (i = 1; i <= (NLon - 1); i++) {
            m           = idx(1, i);
            Node(m, iM) = Node(m, iM) - Node(m, iR2) * (Node(m + NLat, iH) - Node(m, iH));
        }
    }
    if (Imin <= 2) {
        for (j = 1; j <= NLat; j++) {
            m           = idx(j, 1);
            Node(m, iM) = Node(m, iM) - Node(m, iR2) * (Node(m + NLat, iH) - Node(m, iH));
        }
    }
    if (Jmax >= (NLat - 1)) {
        for (i = 1; i <= (NLon - 1); i++) {
            m           = idx(NLat, i);
            Node(m, iM) = Node(m, iM) - Node(m, iR2) * (Node(m + NLat, iH) - Node(m, iH));
        }
    }

// lattitudial flux update
#pragma omp parallel for default(shared) private(i, j, m, v1, v2)
    for (i = Imin; i <= Imax; i++) {
        for (j = Jmin; j <= Jmax; j++) {

            m = idx(j, i);

            if ((Node(m, iD) * Node(m + 1, iD)) == 0)
                continue;

            v1          = Node(m + 1, iH) - Node(m, iH);
            v2          = Node(m - NLat, iM) + Node(m, iM) + Node(m - NLat + 1, iM) + Node(m + 1, iM);
            Node(m, iN) = Node(m, iN) - Node(m, iR4) * v1 - Node(m, iR5) * v2;
        }
    }
    // open boundaries
    if (Imin <= 2) {
        for (j = 1; j <= (NLat - 1); j++) {
            m           = idx(j, 1);
            Node(m, iN) = Node(m, iN) - Node(m, iR4) * (Node(m + 1, iH) - Node(m, iH));
        }
    }
    if (Jmin <= 2) {
        for (i = 1; i <= NLon; i++) {
            m           = idx(1, i);
            Node(m, iN) = Node(m, iN) - Node(m, iR4) * (Node(m + 1, iH) - Node(m, iH));
        }
    }
    if (Imax >= (NLon - 1)) {
        for (j = 1; j <= (NLat - 1); j++) {
            m           = idx(j, NLon);
            Node(m, iN) = Node(m, iN) - Node(m, iR4) * (Node(m + 1, iH) - Node(m, iH));
        }
    }

    // calculation area for the next step
    if (Imin > 2) {
        for (enlarge = 0, j = Jmin; j <= Jmax; j++) {
            if (fabs(Node(idx(j, Imin + 2), iH)) > Par.sshClipThreshold) {
                enlarge = 1;
                break;
            }
        }
        if (enlarge) {
            Imin--;
            if (Imin < 2)
                Imin = 2;
        }
    }
    if (Imax < (NLon - 1)) {
        for (enlarge = 0, j = Jmin; j <= Jmax; j++) {
            if (fabs(Node(idx(j, Imax - 2), iH)) > Par.sshClipThreshold) {
                enlarge = 1;
                break;
            }
        }
        if (enlarge) {
            Imax++;
            if (Imax > (NLon - 1))
                Imax = NLon - 1;
        }
    }
    if (Jmin > 2) {
        for (enlarge = 0, i = Imin; i <= Imax; i++) {
            if (fabs(Node(idx(Jmin + 2, i), iH)) > Par.sshClipThreshold) {
                enlarge = 1;
                break;
            }
        }
        if (enlarge) {
            Jmin--;
            if (Jmin < 2)
                Jmin = 2;
        }
    }
    if (Jmax < (NLat - 1)) {
        for (enlarge = 0, i = Imin; i <= Imax; i++) {
            if (fabs(Node(idx(Jmax - 2, i), iH)) > Par.sshClipThreshold) {
                enlarge = 1;
                break;
            }
        }
        if (enlarge) {
            Jmax++;
            if (Jmax > (NLat - 1))
                Jmax = NLat - 1;
        }
    }

    return 0;
}
