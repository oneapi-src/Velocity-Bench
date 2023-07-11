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
// Created by zeyad-osama on 28/07/2020.
//

#include <operations/components/independents/concrete/model-handlers/SeismicModelHandler.hpp>

#include <operations/utils/io/location_comparator.h>
#include <operations/utils/sampling/Sampler.hpp>
#include <operations/utils/interpolation/Interpolator.hpp>
#include <thoth/api/thoth.hpp>
#include <libraries/nlohmann/json.hpp>
#include <set>

#include <timer/Timer.h>

using namespace std;
using namespace operations::components;
using namespace operations::common;
using namespace operations::dataunits;
using namespace operations::helpers;
using namespace operations::utils::io;
using namespace operations::utils::sampling;
using namespace thoth::streams;
using namespace thoth::dataunits;


SeismicModelHandler::SeismicModelHandler(operations::configuration::ConfigurationMap *apConfigurationMap) {
    this->mpConfigurationMap = apConfigurationMap;
    this->mpGridBox = new GridBox();
};

SeismicModelHandler::~SeismicModelHandler() = default;

void SeismicModelHandler::AcquireConfiguration() {}

void SeismicModelHandler::SetComputationParameters(ComputationParameters *apParameters) {
    this->mpParameters = (ComputationParameters *) apParameters;
    if (this->mpParameters == nullptr) {
        std::cerr << "No computation parameters provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

}

void SeismicModelHandler::SetGridBox(GridBox *apGridBox) {
    this->mpGridBox = apGridBox;
    if (this->mpGridBox == nullptr) {
        std::cerr << "No GridBox provided... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void SeismicModelHandler::SetDependentComponents(
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


GridBox *SeismicModelHandler::ReadModel(map<string, string> const &file_names) {
    this->Initialize(file_names);

    int logical_nx = this->mpGridBox->GetLogicalGridSize(X_AXIS);
    int logical_ny = this->mpGridBox->GetLogicalGridSize(Y_AXIS);
    int logical_nz = this->mpGridBox->GetLogicalGridSize(Z_AXIS);

    int actual_nx = this->mpGridBox->GetActualGridSize(X_AXIS);
    int actual_ny = this->mpGridBox->GetActualGridSize(Y_AXIS);
    int actual_nz = this->mpGridBox->GetActualGridSize(Z_AXIS);

    int initial_nx = this->mpGridBox->GetInitialGridSize(X_AXIS);
    int initial_ny = this->mpGridBox->GetInitialGridSize(Y_AXIS);
    int initial_nz = this->mpGridBox->GetInitialGridSize(Z_AXIS);

    int model_size = actual_nx * actual_ny * actual_nz;
    int initial_size = initial_nx * initial_ny * initial_nz;

    int offset = this->mpParameters->GetBoundaryLength() + this->mpParameters->GetHalfLength();
    int offset_y = actual_ny > 1 ? this->mpParameters->GetBoundaryLength() + this->mpParameters->GetHalfLength() : 0;
    nlohmann::json configuration_map;
    configuration_map[IO_K_PROPERTIES][IO_K_TEXT_HEADERS_ONLY] = false;
    configuration_map[IO_K_PROPERTIES][IO_K_TEXT_HEADERS_STORE] = false;
    thoth::configuration::JSONConfigurationMap io_conf_map(
            configuration_map);
    Reader *seismic_io_reader = new SegyReader(&io_conf_map);
    seismic_io_reader->AcquireConfiguration();

    map<string, float> maximums;
    for (auto const &parameter : this->PARAMS_NAMES) {
        maximums[parameter.second] = 0.0f;
    }
#ifdef ENABLE_GPU_TIMINGS
    Timer *timer = Timer::GetInstance();
#endif
    for (auto const &parameter : this->PARAMS_NAMES) {
        GridBox::Key param_key = parameter.first;
        string param_name = parameter.second;
        vector<Gather *> gathers;
        std::map<std::string, std::string>::const_iterator itFind(file_names.find(param_name)); 
        if (itFind == file_names.end()) {
            ////file_names[param_name].empty()) {
        } else {
#ifdef ENABLE_GPU_TIMINGS
            timer->StartTimer("IO::Read" + param_name + "FromSegyFile");
#endif
            std::vector<TraceHeaderKey> empty_gather_keys;
            std::vector<std::pair<TraceHeaderKey, Gather::SortDirection>> sorting_keys;
            std::vector<std::string> paths = {itFind->second}; ///{file_names[param_name]};
            seismic_io_reader->Initialize(empty_gather_keys, sorting_keys, paths);
            gathers = seismic_io_reader->ReadAll();
#ifdef ENABLE_GPU_TIMINGS
            timer->StopTimer("IO::Read" + param_name + "FromSegyFile");
#endif
        }
        float *parameter_host_buffer = new float[initial_size];
        float *resized_host_buffer = new float[model_size];

        memset(parameter_host_buffer, 0, initial_size * sizeof(float));
        memset(resized_host_buffer, 0, model_size * sizeof(float));

        if (gathers.empty()) {
            cout << "Please provide " << param_name << " model file..." << endl;
            for (unsigned int k = offset_y; k < initial_ny - offset_y; k++) {
                for (unsigned int i = offset; i < initial_nx - offset; i++) {
                    for (unsigned int j = offset; j < initial_nz - offset; j++) {
                        uint index = k * initial_nx * initial_nz + j * initial_nx + i;
                        parameter_host_buffer[index] = 1.0f;
                    }
                }
            }
        } else {
            if (gathers.size() > 1) {
                throw std::runtime_error("Unexpected Number Of Gathers When Reading Parameters : "
                                         + std::to_string(gathers.size()));
            }
            /// sort data
            vector<pair<TraceHeaderKey, Gather::SortDirection>> sorting_keys = {
                    {TraceHeaderKey::SY, Gather::SortDirection::ASC},
                    {TraceHeaderKey::SX, Gather::SortDirection::ASC}
            };
            gathers[0]->SortGather(sorting_keys);


            /// Reading and maximum identification
            for (unsigned int k = offset_y; k < initial_ny - offset_y; k++) {
                for (unsigned int i = offset; i < initial_nx - offset; i++) {
                    for (unsigned int j = offset; j < initial_nz - offset; j++) {

                        uint index = k * initial_nx * initial_nz + j * initial_nx + i;
                        uint trace_index = (k - offset_y) * (initial_nx - 2 * offset) + (i - offset);

                        float temp =
                                parameter_host_buffer[index] =
                                        gathers[0]->GetTrace(trace_index)->GetTraceData()[j - offset];
                        if (temp > maximums[param_name]) {
                            maximums[param_name] = temp;
                        }
                    }
                }
            }
            delete gathers[0];
            gathers.clear();
            seismic_io_reader->Finalize();
        }
        Sampler::Resize(parameter_host_buffer, resized_host_buffer,
                        mpGridBox->GetInitialGridSize(), mpGridBox->GetActualGridSize(),
                        mpParameters);

        auto parameter_ptr = this->mpGridBox->Get(param_key)->GetNativePointer();

        Device::MemCpy(parameter_ptr, resized_host_buffer, sizeof(float) * model_size, Device::COPY_HOST_TO_DEVICE);

        delete[] parameter_host_buffer;
        delete[] resized_host_buffer;
    }

    this->mpGridBox->SetDT(GetSuitableDT(
            this->mpParameters->GetSecondDerivativeFDCoefficient(),
            maximums, this->mpParameters->GetHalfLength(), this->mpParameters->GetRelaxedDT()));

    delete seismic_io_reader;

    return this->mpGridBox;
}

void SeismicModelHandler::Initialize(map<string, string> const &file_names) {
    std::map<std::string, std::string>::const_iterator itFind(file_names.find("velocity"));
    if (itFind == file_names.end()) {
        std::cerr << "Please provide a velocity model... Terminating..." << std::endl;
        exit(EXIT_FAILURE);
    }

#ifdef ENABLE_GPU_TIMINGS
    Timer *timer = Timer::GetInstance();
#endif
    nlohmann::json configuration_map;
    configuration_map[IO_K_PROPERTIES][IO_K_TEXT_HEADERS_ONLY] = false;
    configuration_map[IO_K_PROPERTIES][IO_K_TEXT_HEADERS_STORE] = false;
    thoth::configuration::JSONConfigurationMap io_conf_map(
            configuration_map);
    Reader *seismic_io_reader = new SegyReader(&io_conf_map);

#ifdef ENABLE_GPU_TIMINGS
    timer->StartTimer("IO::ReadVelocityFromSegyFile");
#endif
    std::vector<TraceHeaderKey> empty_gather_keys;
    std::vector<std::pair<TraceHeaderKey, Gather::SortDirection>> sorting_keys;
    std::vector<std::string> paths = {itFind->second};
    seismic_io_reader->Initialize(empty_gather_keys, sorting_keys,
                                  paths);
    vector<Gather *> gathers;
    gathers = seismic_io_reader->ReadAll();
    auto gather = gathers[0];
#ifdef ENABLE_GPU_TIMINGS
    timer->StopTimer("IO::ReadVelocityFromSegyFile");
#endif
    set<float> x_locations;
    set<float> y_locations;
    for (uint i = 0; i < gather->GetNumberTraces(); i++) {
        x_locations.emplace(
                gather->GetTrace(i)->GetTraceHeaderKeyValue<float>(TraceHeaderKey::SX));
        y_locations.emplace(
                gather->GetTrace(i)->GetTraceHeaderKeyValue<float>(TraceHeaderKey::SY));
    }
    int nx, ny, nz;
    nx = x_locations.size() +
         2 * this->mpParameters->GetBoundaryLength() +
         2 * this->mpParameters->GetHalfLength();
    this->mpGridBox->SetLogicalGridSize(X_AXIS, nx);
    this->mpGridBox->SetInitialGridSize(X_AXIS, nx);

    nz = gather->GetTrace(0)->GetNumberOfSamples() +
         2 * this->mpParameters->GetBoundaryLength() +
         2 * this->mpParameters->GetHalfLength();
    this->mpGridBox->SetLogicalGridSize(Z_AXIS, nz);
    this->mpGridBox->SetInitialGridSize(Z_AXIS, nz);

    if (y_locations.size() > 1) {
        ny = y_locations.size() +
             2 * this->mpParameters->GetBoundaryLength() +
             2 * this->mpParameters->GetHalfLength();
    } else {
        ny = 1;
    }
    this->mpGridBox->SetLogicalGridSize(Y_AXIS, ny);
    this->mpGridBox->SetInitialGridSize(Y_AXIS, ny);

    float dx, dy, dz, dt;

    dx = *(++x_locations.begin()) - (*x_locations.begin());
    if (y_locations.size() > 1) {
        dy = *(++y_locations.begin()) - (*y_locations.begin());
    } else {
        dy = 0;
    }
    // For some reason, bp has this unit in centimeters, not meter violating standard.
    // Some scalar must be there to adjust it, until it's found, placeholder.
    //TODO fix correct scaling for space sampling.
    dz = gather->GetSamplingRate() / 1e3;

    this->mpGridBox->SetInitialCellDimensions(X_AXIS, dx);
    this->mpGridBox->SetInitialCellDimensions(Y_AXIS, dy);
    this->mpGridBox->SetInitialCellDimensions(Z_AXIS, dz);

    int last = gather->GetNumberTraces() - 1;


    this->mpGridBox->SetReferencePoint(X_AXIS,
                                       std::min(
                                               gather->GetTrace(0)->GetTraceHeaderKeyValue<float>(
                                                       TraceHeaderKey::SX),
                                               gather->GetTrace(last)->GetTraceHeaderKeyValue<float>(
                                                       TraceHeaderKey::SX)));

    this->mpGridBox->SetReferencePoint(Z_AXIS, 0);

    this->mpGridBox->SetReferencePoint(Y_AXIS,
                                       std::min(
                                               gather->GetTrace(0)->GetTraceHeaderKeyValue<float>(
                                                       TraceHeaderKey::SY),
                                               gather->GetTrace(last)->GetTraceHeaderKeyValue<float>(
                                                       TraceHeaderKey::SY)));

    cout << "Reference X\t: " << this->mpGridBox->GetReferencePoint(X_AXIS) << endl
         << "Reference Y\t: " << this->mpGridBox->GetReferencePoint(Y_AXIS) << endl
         << "Reference Z\t: " << this->mpGridBox->GetReferencePoint(Z_AXIS) << endl;

    /// If there is no window then window size equals full model size
    if (this->mpParameters->IsUsingWindow()) {
        this->mpGridBox->SetWindowStart(X_AXIS, 0);
        this->mpGridBox->SetWindowStart(Y_AXIS, 0);
        this->mpGridBox->SetWindowStart(Z_AXIS, 0);

        if (this->mpParameters->GetLeftWindow() == 0 && this->mpParameters->GetRightWindow() == 0) {
            this->mpGridBox->SetLogicalWindowSize(X_AXIS, nx);
        } else {
            this->mpGridBox->SetLogicalWindowSize(X_AXIS, std::min(
                    this->mpParameters->GetLeftWindow() +
                    this->mpParameters->GetRightWindow() +
                    1 +
                    2 * this->mpParameters->GetBoundaryLength() +
                    2 * this->mpParameters->GetHalfLength(), nx));
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
                    this->mpParameters->GetBackWindow() + 1 +
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

    // Assume minimum velocity to be 1500 m/s for now.
    Sampler::CalculateAdaptiveCellDimensions(mpGridBox, mpParameters, 1500);

    this->SetupPadding();

    this->RegisterWaveFields(nx, ny, nz);
    this->RegisterParameters(nx, ny, nz);

    this->AllocateWaveFields();
    this->AllocateParameters();
    seismic_io_reader->Finalize();
    delete seismic_io_reader;
}

void SeismicModelHandler::RegisterWaveFields(uint nx, uint ny, uint nz) {
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

void SeismicModelHandler::RegisterParameters(uint nx, uint ny, uint nz) {
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

void SeismicModelHandler::AllocateWaveFields() {
    uint wnx = this->mpGridBox->GetActualWindowSize(X_AXIS);
    uint wny = this->mpGridBox->GetActualWindowSize(Y_AXIS);
    uint wnz = this->mpGridBox->GetActualWindowSize(Z_AXIS);
    uint window_size = wnx * wny * wnz;

    /// Allocating and zeroing wave fields.
    for (auto wave_field : this->WAVE_FIELDS_NAMES) {
        if (!GridBox::Includes(wave_field, NEXT)) {

            auto frame_buffer = new FrameBuffer<float>();
            frame_buffer->Allocate(window_size,
                                   mpParameters->GetHalfLength(),
                                   GridBox::Stringify(wave_field));

            this->mpWaveFieldsMemoryHandler->FirstTouch(frame_buffer->GetNativePointer(), this->mpGridBox, true);
            this->mpGridBox->RegisterWaveField(wave_field, frame_buffer);

            if (GridBox::Includes(wave_field, GB_PRSS | PREV)) {
                GridBox::Replace(&wave_field, PREV, NEXT);
                this->mpGridBox->RegisterWaveField(wave_field, frame_buffer);
            } else if (this->mpParameters->GetEquationOrder() == FIRST &&
                       GridBox::Includes(wave_field, GB_PRSS | CURR)) {
                GridBox::Replace(&wave_field, CURR, NEXT);
                this->mpGridBox->RegisterWaveField(wave_field, frame_buffer);
            }
        }
    }
}

void SeismicModelHandler::AllocateParameters() {
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


            auto frame_buffer = new FrameBuffer<float>();
            frame_buffer->Allocate(grid_size,
                                   mpParameters->GetHalfLength(),
                                   param_name);

            this->mpWaveFieldsMemoryHandler->FirstTouch(frame_buffer->GetNativePointer(), this->mpGridBox);

            auto frame_buffer_window = new FrameBuffer<float>();
            frame_buffer_window->Allocate(window_size,
                                          mpParameters->GetHalfLength(),
                                          param_name);

            this->mpWaveFieldsMemoryHandler->FirstTouch(frame_buffer_window->GetNativePointer(), this->mpGridBox, true);

            this->mpGridBox->RegisterParameter(param_key, frame_buffer, frame_buffer_window);
        }
    } else {
        for (auto const &parameter : this->PARAMS_NAMES) {
            GridBox::Key param_key = parameter.first;
            string param_name = parameter.second;

            auto frame_buffer = new FrameBuffer<float>();
            frame_buffer->Allocate(grid_size,
                                   mpParameters->GetHalfLength(),
                                   param_name);

            this->mpWaveFieldsMemoryHandler->FirstTouch(frame_buffer->GetNativePointer(), this->mpGridBox);

            this->mpGridBox->RegisterParameter(param_key, frame_buffer);
        }
    }
}

float SeismicModelHandler::GetSuitableDT
        (float *coefficients, map<string, float> maximums, int half_length, float dt_relax) {
    float dx = this->mpGridBox->GetCellDimensions(X_AXIS);
    float dy = this->mpGridBox->GetCellDimensions(Y_AXIS);
    float dz = this->mpGridBox->GetCellDimensions(Z_AXIS);

    uint ny = this->mpGridBox->GetLogicalGridSize(Y_AXIS);

    // Calculate dt through finite difference stability equation
    float dxSquare = 1 / (dx * dx);
    float dySquare;

    if (ny != 1)
        dySquare = 1 / (dy * dy);
    else // case of 2D
        dySquare = 0.0;

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
