/*
 * Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU Lesser General Public License v3.0 or later
 * 
 * If a copy of the license was not distributed with this file, you can obtain one at 
 * https://www.gnu.org/licenses/lgpl-3.0-standalone.html
 * 
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


#ifndef MANIPULATION_h
#define MANIPULATION_h

#include <../datatypes.h>

class Manipulation {

public:
    // sotring trace form type to another , exapmle from csr to cmp or vice versa
    // , his a high level function that can call nested function that will do the
    // sorting
    void TracesSort(vector<genral_traces> traces_in,
                    vector<genral_traces> traces_sorted, string sort_from,
                    string sort_to);

    void TracesToMatrix(vector<genral_traces> traces, float *data_matrix);
}

#endif MANIPULATION_h