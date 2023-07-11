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


//
// Created by amr-nasr on 04/07/2020.
//

#include <operations/utils/io/read_utils.h>


#include <memory-manager/MemoryManager.h>

#include <unordered_set>
#include <algorithm>

using namespace operations::utils::io;
using namespace operations::dataunits;
using namespace operations::common;
using namespace thoth::dataunits;


void operations::utils::io::ParseGatherToTraces(
        thoth::dataunits::Gather *apGather, Point3D *apSource, TracesHolder *apTraces,
        uint **x_position, uint **y_position,
        GridBox *apGridBox, ComputationParameters *apParameters,
        float *total_time) {
    std::unordered_set<uint> x_dim;
    std::unordered_set<uint> y_dim;
    // No need to sort as we deal with each trace with its position independently.
//    vector<pair<TraceHeaderKey, Gather::SortDirection>> sorting_keys = {
//            {TraceHeaderKey::GY, Gather::SortDirection::ASC},
//            {TraceHeaderKey::GX, Gather::SortDirection::ASC}
//    };
//    apGather->SortGather(sorting_keys);

    // Get source point.
    apSource->x = (apGather->GetTrace(0)->GetTraceHeaderKeyValue<float>(TraceHeaderKey::SX)
                   - apGridBox->GetReferencePoint(X_AXIS)) /
                  (apGridBox->GetCellDimensions(X_AXIS));
    apSource->z = 0;
    apSource->y = (apGather->GetTrace(0)->GetTraceHeaderKeyValue<float>(TraceHeaderKey::SY)
                   - apGridBox->GetReferencePoint(Y_AXIS)) /
                  (apGridBox->GetCellDimensions(Y_AXIS));
    // If window model, need to setup the starting point of the window and adjust source point.
    // Handle 3 cases : no room for left window, no room for right window, room for both.
    // Those 3 cases can apply to y-direction as well if 3D.
    if (apParameters->IsUsingWindow()) {
        apGridBox->SetWindowStart(X_AXIS, 0);
        // No room for left window.
        if (apSource->x < apParameters->GetLeftWindow() ||
            (apParameters->GetLeftWindow() == 0 && apParameters->GetRightWindow() == 0)) {
            apGridBox->SetWindowStart(X_AXIS, 0);
            // No room for right window.
        } else if (apSource->x >= apGridBox->GetLogicalGridSize(X_AXIS)
                                  - apParameters->GetBoundaryLength() - apParameters->GetHalfLength() -
                                  apParameters->GetRightWindow()) {
            apGridBox->SetWindowStart(X_AXIS, apGridBox->GetLogicalGridSize(X_AXIS) -
                                              apParameters->GetBoundaryLength() -
                                              apParameters->GetHalfLength() -
                                              apParameters->GetRightWindow() -
                                              apParameters->GetLeftWindow() - 1);
            apSource->x = apGridBox->GetLogicalWindowSize(X_AXIS) -
                          apParameters->GetBoundaryLength() -
                          apParameters->GetHalfLength() -
                          apParameters->GetRightWindow() - 1;
        } else {
            apGridBox->SetWindowStart(X_AXIS, apSource->x - apParameters->GetLeftWindow());
            apSource->x = apParameters->GetLeftWindow();
        }
        apGridBox->SetWindowStart(Y_AXIS, 0);
        if (apGridBox->GetLogicalGridSize(Y_AXIS) != 1) {
            if (apSource->y < apParameters->GetBackWindow() ||
                (apParameters->GetFrontWindow() == 0 && apParameters->GetBackWindow() == 0)) {
                apGridBox->SetWindowStart(Y_AXIS, 0);
            } else if (apSource->y >= apGridBox->GetLogicalGridSize(Y_AXIS)
                                      - apParameters->GetBoundaryLength() - apParameters->GetHalfLength() -
                                      apParameters->GetFrontWindow()) {
                apGridBox->SetWindowStart(Y_AXIS, apGridBox->GetLogicalGridSize(Y_AXIS) -
                                                  apParameters->GetBoundaryLength() -
                                                  apParameters->GetHalfLength() -
                                                  apParameters->GetFrontWindow() -
                                                  apParameters->GetBackWindow() - 1);
                apSource->y = apGridBox->GetWindowStart(Y_AXIS) -
                              apParameters->GetBoundaryLength() -
                              apParameters->GetHalfLength() -
                              apParameters->GetFrontWindow() - 1;
            } else {
                apGridBox->SetWindowStart(Y_AXIS, apSource->y - apParameters->GetBackWindow());
                apSource->y = apParameters->GetBackWindow();
            }
        }
    }
    // Remove traces outside the window.
    int intern_x = apGridBox->GetLogicalWindowSize(X_AXIS) - 2 * apParameters->GetHalfLength() -
                   2 * apParameters->GetBoundaryLength();
    int intern_y = apGridBox->GetLogicalWindowSize(Y_AXIS) - 2 * apParameters->GetHalfLength() -
                   2 * apParameters->GetBoundaryLength();
    for (int i = ((int) apGather->GetNumberTraces()) - 1; i >= 0; i--) {
        bool erased = false;
        int gx = (apGather->GetTrace(i)->GetTraceHeaderKeyValue<float>(TraceHeaderKey::GX)
                  - apGridBox->GetReferencePoint(X_AXIS)) /
                 (apGridBox->GetCellDimensions(X_AXIS));
        int gy = (apGather->GetTrace(i)->GetTraceHeaderKeyValue<float>(TraceHeaderKey::GY)
                  - apGridBox->GetReferencePoint(Y_AXIS)) /
                 (apGridBox->GetCellDimensions(Y_AXIS));
        if (gx < apGridBox->GetWindowStart(X_AXIS) || gx >= apGridBox->GetWindowStart(X_AXIS) + intern_x) {
            apGather->RemoveTrace(i);
            erased = true;
        } else if (apGridBox->GetLogicalGridSize(Y_AXIS) != 1) {
            if (gy < apGridBox->GetWindowStart(Y_AXIS) || gy >= apGridBox->GetWindowStart(Y_AXIS) + intern_y) {
                apGather->RemoveTrace(i);
                erased = true;
            }
        }

        if (!erased) {
            x_dim.insert(apGather->GetTrace(i)->GetTraceHeaderKeyValue<float>(TraceHeaderKey::GX));
            y_dim.insert(apGather->GetTrace(i)->GetTraceHeaderKeyValue<float>(TraceHeaderKey::GY));
        }
    }
    // Set meta data.
    apTraces->SampleDT = apGather->GetSamplingRate() / (float) 1e6;
    int sample_nt = apGather->GetTrace(0)->GetNumberOfSamples();

    int num_elements_per_time_step = apGather->GetNumberTraces();
    apTraces->TraceSizePerTimeStep = apGather->GetNumberTraces();
    apTraces->ReceiversCountX = x_dim.size();
    apTraces->ReceiversCountY = y_dim.size();
    apTraces->SampleNT = sample_nt;

    /// We dont have total time , but we have the nt from the
    /// segy file , so we can modify the nt according to the
    /// ratio between the recorded dt and the suitable dt
    apGridBox->SetNT(int(sample_nt * apTraces->SampleDT / apGridBox->GetDT()));

    *total_time = sample_nt * apTraces->SampleDT;
    // Setup traces data to the arrays.
    apTraces->Traces = (float *) mem_allocate(
            sizeof(float), sample_nt * num_elements_per_time_step, "traces");
    *x_position = (uint *) mem_allocate(
            sizeof(uint), num_elements_per_time_step, "traces x-position");
    *y_position = (uint *) mem_allocate(
            sizeof(uint), num_elements_per_time_step, "traces y-position");

    for (int trace_index = 0; trace_index < num_elements_per_time_step; trace_index++) {
        for (int t = 0; t < sample_nt; t++) {
            apTraces->Traces[t * num_elements_per_time_step + trace_index] =
                    apGather->GetTrace(trace_index)->GetTraceData()[t];
        }
        int gx = (apGather->GetTrace(trace_index)->GetTraceHeaderKeyValue<float>(TraceHeaderKey::GX)
                  - apGridBox->GetReferencePoint(X_AXIS)) /
                 (apGridBox->GetCellDimensions(X_AXIS));
        int gy = (apGather->GetTrace(trace_index)->GetTraceHeaderKeyValue<float>(TraceHeaderKey::GY)
                  - apGridBox->GetReferencePoint(Y_AXIS)) /
                 (apGridBox->GetCellDimensions(Y_AXIS));
        gx -= apGridBox->GetWindowStart(X_AXIS);
        gy -= apGridBox->GetWindowStart(Y_AXIS);
        (*x_position)[trace_index] = gx + apParameters->GetHalfLength() + apParameters->GetBoundaryLength();
        if (apGridBox->GetLogicalGridSize(Y_AXIS) > 1) {
            (*y_position)[trace_index] = gy + apParameters->GetHalfLength() + apParameters->GetBoundaryLength();
        } else {
            (*y_position)[trace_index] = 0;
        }
    }
}
