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
// Created by mirna-moawad on 22/10/2019.
//

#include "operations/components/independents/concrete/model-handlers/SyntheticModelHandler.hpp"

#include <iostream>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;
using namespace operations::helpers;

SyntheticModelHandler::SyntheticModelHandler(operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mpGridBox = new GridBox();
}

SyntheticModelHandler::~SyntheticModelHandler() = default;

void SyntheticModelHandler::AcquireConfiguration() {}

void SyntheticModelHandler::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void SyntheticModelHandler::SetGridBox(GridBox *apGridBox) {
    this->mpGridBox = apGridBox;
    if (this->mpGridBox == nullptr) {
        std::cerr << "No GridBox provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void SyntheticModelHandler::SetDependentComponents(
        ComponentsMap<DependentComponent> *apDependentComponentsMap) {
    HasDependents::SetDependentComponents(apDependentComponentsMap);

    this->mpWaveFieldsMemoryHandler =
            (WaveFieldsMemoryHandler *)
                    this->GetDependentComponentsMap()->Get(MEMORY_HANDLER);
    if (this->mpWaveFieldsMemoryHandler == nullptr) {
        std::cerr << "No Wave Fields Memory Handler provided... "
                  << "Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

GridBox *SyntheticModelHandler::ReadModel(map<string, string> const &file_names) {
    this->Initialize(file_names);

    int nx = this->mpGridBox->GetActualGridSize(X_AXIS);
    int ny = this->mpGridBox->GetActualGridSize(Y_AXIS);
    int nz = this->mpGridBox->GetActualGridSize(Z_AXIS);
    int logical_nx = this->mpGridBox->GetLogicalGridSize(X_AXIS);
    int logical_ny = this->mpGridBox->GetLogicalGridSize(Y_AXIS);
    int logical_nz = this->mpGridBox->GetLogicalGridSize(Z_AXIS);

    map<string, float> maximums;
    for (auto const &parameter : this->PARAMS_NAMES) {
        maximums[parameter.second] = 0.0f;
    }

    for (auto const &parameter : this->PARAMS_NAMES) {
        GridBox::Key param_key = parameter.first;
        string param_name = parameter.second;
        /// @todo this->mModelFile[index] ???
        maximums[param_name] =
                this->SetModelField(this->mpGridBox->Get(param_key)->GetNativePointer(),
                                    this->mModelFile[1], nx, nz, ny, logical_nx,
                                    logical_nz, logical_ny);
    }

    this->mpGridBox->SetDT(GetSuitableDT(
            this->mpParameters->GetSecondDerivativeFDCoefficient(),
            maximums, this->mpParameters->GetHalfLength(), this->mpParameters->GetRelaxedDT()));

    return this->mpGridBox;
}

void SyntheticModelHandler::Initialize(map<string, string> const &file_names) {

    std::map<std::string, std::string>::const_iterator itFind(file_names.find("velocity"));
    if (itFind == file_names.end()) {
        std::cerr << "Please provide a velocity model... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string const file_name(itFind->second); // file_names.find("velocity").second; ///file_names["velocity"];
    ifstream inFile(file_name);
    string temp_line;
    string temp_s;
    float temp;
    vector<string> v;
    // Read each line in the text file and insert in vector V
    while (getline(inFile, temp_line, '|')) {
        v.push_back(temp_line);
    }
    // Loop on vector v and insert nx,ny,nz,dx,dy,dz in this->mModelFile[0]
    // velocity, start and end of each layer in this->mModelFile[1];
    // start and end of window in this->mModelFile[2]
    for (int i = 0; i < v.size(); i++) {
        stringstream ss(v[i]);
        string item;
        vector<float> items;
        while (getline(ss, item, ',')) {
            items.push_back(atof(item.c_str()));
        }
        this->mModelFile.push_back(items);
    }

    /// Set grid size
    int nx, ny, nz;
    nx = this->mModelFile[0][0] + 2 * this->mpParameters->GetBoundaryLength() + 2 * this->mpParameters->GetHalfLength();
    nz = this->mModelFile[0][1] + 2 * this->mpParameters->GetBoundaryLength() + 2 * this->mpParameters->GetHalfLength();
    this->mpGridBox->SetLogicalGridSize(X_AXIS, nx);
    this->mpGridBox->SetLogicalGridSize(Z_AXIS, nz);
    if (this->mModelFile[0][2] > 1) {
        ny = this->mModelFile[0][2] +
             2 * this->mpParameters->GetBoundaryLength() +
             2 * this->mpParameters->GetHalfLength();
    } else {
        ny = 1;
    }
    this->mpGridBox->SetLogicalGridSize(Y_AXIS, ny);

    /// Set cell dimensions
    float dx, dy, dz, dt;
    dx = this->mModelFile[0][3];
    dz = this->mModelFile[0][4];
    dy = this->mModelFile[0][5];
    this->mpGridBox->SetCellDimensions(X_AXIS, dx);
    this->mpGridBox->SetCellDimensions(Y_AXIS, dy);
    this->mpGridBox->SetCellDimensions(Z_AXIS, dz);

    unsigned int model_size = nx * nz * ny;

    // if there is no window then window size equals full model size
    if (this->mpParameters->IsUsingWindow()) {
        this->mpGridBox->SetWindowStart(X_AXIS, 0);
        this->mpGridBox->SetWindowStart(Y_AXIS, 0);
        this->mpGridBox->SetWindowStart(Z_AXIS, 0);

        if (this->mpParameters->GetLeftWindow() == 0 &&
            this->mpParameters->GetRightWindow() == 0) {
            this->mpGridBox->SetLogicalWindowSize(X_AXIS, nx);
        } else {
            this->mpGridBox->SetLogicalWindowSize(X_AXIS, std::min(
                    this->mpParameters->GetLeftWindow() +
                    this->mpParameters->GetRightWindow() +
                    1 +
                    2 * this->mpParameters->GetBoundaryLength() +
                    2 * this->mpParameters->GetHalfLength(),
                    nx));
        }
        if (this->mpParameters->GetDepthWindow() == 0) {
            this->mpGridBox->SetLogicalWindowSize(Z_AXIS, nz);
        } else {
            this->mpGridBox->SetLogicalWindowSize(Z_AXIS, std::min(
                    this->mpParameters->GetDepthWindow() +
                    2 * this->mpParameters->GetBoundaryLength() +
                    2 * this->mpParameters->GetHalfLength(), nz));
        }
        if ((this->mpParameters->GetFrontWindow() == 0 &&
             this->mpParameters->GetBackWindow() == 0) ||
            ny == 1) {
            this->mpGridBox->SetLogicalWindowSize(Y_AXIS, ny);
        } else {
            this->mpGridBox->SetLogicalWindowSize(Y_AXIS, std::min(
                    this->mpParameters->GetFrontWindow() +
                    this->mpParameters->GetBackWindow() +
                    1 +
                    2 * this->mpParameters->GetBoundaryLength() +
                    2 * this->mpParameters->GetHalfLength(), ny));
        }
    } else {
        this->mpGridBox->SetWindowStart(X_AXIS, 0);
        this->mpGridBox->SetWindowStart(Y_AXIS, 0);
        this->mpGridBox->SetWindowStart(Z_AXIS, 0);

        this->mpGridBox->SetLogicalWindowSize(X_AXIS, nx);
        this->mpGridBox->SetLogicalWindowSize(Y_AXIS, ny);
        this->mpGridBox->SetLogicalWindowSize(Z_AXIS, nz);
    }
    this->SetupPadding();

    this->RegisterWaveFields(nx, ny, nx);
    this->RegisterParameters(nx, ny, nx);

    this->AllocateWaveFields();
    this->AllocateParameters();
}


void SyntheticModelHandler::RegisterWaveFields(uint nx, uint ny, uint nz) {
    /// Register wave field by order
    if (this->mpParameters->GetEquationOrder() == FIRST) {
        this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRTC | CURR | DIR_Z);
        this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRTC | CURR | DIR_X);
        if (ny > 1) {
            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRTC | CURR | DIR_Y);
        }
        /// Register wave field by approximation
        if (this->mpParameters->GetApproximation() == ISOTROPIC) {
            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRSS | CURR);
            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRSS | NEXT);
        }
    } else if (this->mpParameters->GetEquationOrder() == SECOND) {
        /// Register wave field by approximation
        if (this->mpParameters->GetApproximation() == ISOTROPIC) {
            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRSS | CURR);
            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRSS | PREV);
            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRSS | NEXT);
        } else if (this->mpParameters->GetApproximation() == VTI ||
                   this->mpParameters->GetApproximation() == TTI) {
            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRSS | CURR | DIR_Z);
            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRSS | PREV | DIR_Z);
            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRSS | NEXT | DIR_Z);

            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRSS | CURR | DIR_X);
            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRSS | PREV | DIR_X);
            this->WAVE_FIELDS_NAMES.push_back(WAVE | GB_PRSS | NEXT | DIR_X);
        }
    }
}

void SyntheticModelHandler::RegisterParameters(uint nx, uint ny, uint nz) {
    /// Register parameters by order
    this->PARAMS_NAMES.push_back(std::make_pair(PARM | GB_VEL, "velocity"));
    if (this->mpParameters->GetEquationOrder() == FIRST) {
        this->PARAMS_NAMES.push_back(std::make_pair(PARM | GB_DEN, "density"));
    }
    if (this->mpParameters->GetApproximation() == VTI ||
        this->mpParameters->GetApproximation() == TTI) {
        this->PARAMS_NAMES.push_back(std::make_pair(PARM | GB_DLT, "delta"));
        this->PARAMS_NAMES.push_back(std::make_pair(PARM | GB_EPS, "epsilon"));
    }
    if (this->mpParameters->GetApproximation() == TTI) {
        this->PARAMS_NAMES.push_back(std::make_pair(PARM | GB_THT, "theta"));
        this->PARAMS_NAMES.push_back(std::make_pair(PARM | GB_PHI, "phi"));
    }
}

void SyntheticModelHandler::AllocateWaveFields() {
    uint wnx = this->mpGridBox->GetActualWindowSize(X_AXIS);
    uint wny = this->mpGridBox->GetActualWindowSize(Y_AXIS);
    uint wnz = this->mpGridBox->GetActualWindowSize(Z_AXIS);
    uint window_size = wnx * wny * wnz;

    /// Allocating and zeroing wave fields.
    for (auto wave_field : this->WAVE_FIELDS_NAMES) {
        if (!GridBox::Includes(wave_field, NEXT)) {

            FrameBuffer<float> *wave_field_buffer = new FrameBuffer<float>();
            wave_field_buffer->Allocate(window_size, mpParameters->GetHalfLength(), GridBox::Stringify(wave_field));

            this->mpWaveFieldsMemoryHandler->FirstTouch(wave_field_buffer->GetNativePointer(), this->mpGridBox, true);

            this->mpGridBox->RegisterWaveField(wave_field, wave_field_buffer);

            if (GridBox::Includes(wave_field, GB_PRSS | PREV)) {
                GridBox::Replace(&wave_field, PREV, NEXT);
                this->mpGridBox->RegisterWaveField(wave_field, wave_field_buffer);
            } else if (this->mpParameters->GetEquationOrder() == FIRST &&
                       GridBox::Includes(wave_field, GB_PRSS | CURR)) {
                GridBox::Replace(&wave_field, CURR, NEXT);
                this->mpGridBox->RegisterWaveField(wave_field, wave_field_buffer);
            }
        }
    }
}

void SyntheticModelHandler::AllocateParameters() {
    uint nx = this->mpGridBox->GetActualGridSize(X_AXIS);
    uint ny = this->mpGridBox->GetActualGridSize(Y_AXIS);
    uint nz = this->mpGridBox->GetActualGridSize(Z_AXIS);
    uint grid_size = nx * nz * ny;

    uint wnx = this->mpGridBox->GetActualWindowSize(X_AXIS);
    uint wny = this->mpGridBox->GetActualWindowSize(Y_AXIS);
    uint wnz = this->mpGridBox->GetActualWindowSize(Z_AXIS);
    uint window_size = wnx * wnz * wny;

    if (this->mpParameters->IsUsingWindow()) {
        for (auto const &parameter : this->PARAMS_NAMES) {
            GridBox::Key param_key = parameter.first;
            string param_name = parameter.second;

            FrameBuffer<float> *frame_buffer = new FrameBuffer<float>();
            frame_buffer->Allocate(grid_size, mpParameters->GetHalfLength(), param_name);

            this->mpWaveFieldsMemoryHandler->FirstTouch(frame_buffer->GetNativePointer(), this->mpGridBox);

            FrameBuffer<float> *frame_buffer_window = new FrameBuffer<float>();
            frame_buffer_window->Allocate(window_size, mpParameters->GetHalfLength(), "window_" + param_name);

            this->mpWaveFieldsMemoryHandler->FirstTouch(frame_buffer_window->GetNativePointer(), this->mpGridBox, true);

            this->mpGridBox->RegisterParameter(param_key, frame_buffer, frame_buffer_window);
        }
    } else {
        for (auto const &parameter : this->PARAMS_NAMES) {
            GridBox::Key param_key = parameter.first;
            string param_name = parameter.second;

            FrameBuffer<float> *frame_buffer = new FrameBuffer<float>();
            frame_buffer->Allocate(grid_size, mpParameters->GetHalfLength(), param_name);;
            this->mpWaveFieldsMemoryHandler->FirstTouch(frame_buffer->GetNativePointer(), this->mpGridBox);

            this->mpGridBox->RegisterParameter(param_key, frame_buffer);
        }
    }
}

float SyntheticModelHandler::GetSuitableDT(
        float *coefficients, map<string, float> maximums, int half_length, float dt_relax) {

    int ny = this->mpGridBox->GetLogicalGridSize(Y_AXIS);

    float dx = this->mpGridBox->GetCellDimensions(X_AXIS);
    float dy = this->mpGridBox->GetCellDimensions(Y_AXIS);
    float dz = this->mpGridBox->GetCellDimensions(Z_AXIS);

    // Calculate dt through finite difference stability equation
    float dxSquare = 1 / (dx * dx);
    float dySquare;

    if (ny != 1) {
        dySquare = 1 / (dy * dy);
    } else {
        dySquare = 0.0;
    }

    float dzSquare = 1 / (dz * dz);

    float distanceM = 1 / (sqrtf(dxSquare + dySquare + dzSquare));

    /// The sum of absolute value of weights for second derivative if
    /// du per dt. we use second order so the weights are (-1,2,1)
    float a1 = 4;
    float a2 = 0;

    for (int i = 1; i <= half_length; i++) {
        a2 += fabs(coefficients[i]);
    }
    a2 *= 2.0;
    /// The sum of absolute values for second derivative id
    /// du per dx ( one dimension only )
    a2 += fabs(coefficients[0]);

    if (this->mpParameters->GetApproximation() == VTI) {
        a2 *= (2 + 4 * maximums["epsilon"] + sqrtf(1 + 2 * maximums["delta"]));
    } else if (this->mpParameters->GetApproximation() == TTI) {
        float tti_coefficients = (powf(cos(maximums["theta"]), 2) * sin(2 * maximums["phi"])) +
                                 (sin(2 * maximums["theta"]) * (sin(maximums["phi"]) + cos(maximums["phi"])));
        float b1 = (1 + 2 * maximums["epsilon"]) * (2 - tti_coefficients);
        float b2 = sqrtf(1 + 2 * maximums["delta"]);
        a2 *= (b1 + b2);
    }
    float dt = ((sqrtf(a1 / a2)) * distanceM) / maximums["velocity"] * dt_relax;
    return dt;
}
