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
// Created by pancee on 12/28/20.
//

#include <thoth/utils/synthetic-generators/SyntheticModelGenerator.hpp>

#include <thoth/configurations/interface/MapKeys.h>

#include <libraries/nlohmann/json.hpp>

#include <iostream>
#include <random>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

using namespace std;
using namespace thoth;
using namespace thoth::generators;
using namespace thoth::dataunits;
using json = nlohmann::json;


SyntheticModelGenerator::SyntheticModelGenerator(thoth::configuration::ConfigurationMap *apConfigurationMap)
        : mModel() {
    this->mpConfigurationMap = apConfigurationMap;
    this->GeneratedTracesData = nullptr;
}

SyntheticModelGenerator::~SyntheticModelGenerator() {
    if (this->GeneratedTracesData) {
        free(this->GeneratedTracesData);
    }
}

void SyntheticModelGenerator::AcquireConfiguration() {
    this->mMetaDataFilepath = this->mpConfigurationMap->GetValue(
            IO_K_PROPERTIES, IO_K_READ_PATH, this->mMetaDataFilepath);
}

void SyntheticModelGenerator::BuildGather() {
    uint nx = 0, ny = 0, nz = 0, dx = 0;
    float sx = 0, gx = 0;

    nx = this->mMetaData["grid-size"]["nx"].get<uint>();
    nz = this->mMetaData["grid-size"]["nz"].get<uint>();

    dx = this->mMetaData["cell-dimension"]["dx"].get<float>();

    TraceHeaderKey trace_header_sx(TraceHeaderKey::SX);
    TraceHeaderKey trace_header_gx(TraceHeaderKey::GX);
    std::vector<Trace *> traces(nx);

    for (int ix = 0; ix < nx; ++ix) {
        traces.at(ix) = new Trace(nz);
    }

    for (int ix = 0; ix < nx; ++ix) {
        auto trace = traces.at(ix);
        auto p_trace_data = trace->GetTraceData();
        /// UPDATE THIS PART TO CALCULATE SX/GX AND SY/GY ACCORDING TO MODEL OFFSET_TYPE
        /// HERE IT CALCULATES SX = GX AS THE MODEL IS VELOCITY ONE
        /// NO SY/GY AS IT IS A 2D ONE
        sx = gx = ix * dx;
        trace->SetTraceHeaderKeyValue<float>(trace_header_sx, sx);
        trace->SetTraceHeaderKeyValue<float>(trace_header_gx, gx);
        for (int iz = 0; iz < nz; ++iz) {
            p_trace_data[iz] = this->GeneratedTracesData[iz * nx + ix];
        }
        this->mModel.AddTrace(trace);
    }
    free(this->GeneratedTracesData);
    this->GeneratedTracesData = nullptr;
}

void SyntheticModelGenerator::Generate() {
    std::ifstream in(this->mMetaDataFilepath);
    in >> this->mMetaData;

    uint nx = 0, ny = 0, nz = 0;
    nx = this->mMetaData["grid-size"]["nx"].get<uint>();
    ny = this->mMetaData["grid-size"]["ny"].get<uint>();
    nz = this->mMetaData["grid-size"]["nz"].get<uint>();

    uint bytes = nx * ny * nz * sizeof(float);
    this->GeneratedTracesData = (float *) malloc(bytes);
    memset(this->GeneratedTracesData, 0.0f, bytes);

    this->GenerateModel();
    cout << "Synthetic model generated successfully...";
}

void SyntheticModelGenerator::GenerateModel() {
    this->GenerateLayers();
    this->InjectSaltBodies();
    this->InjectCracks();
}

void SyntheticModelGenerator::GenerateLayers() {
    uint nx = this->mMetaData["grid-size"]["nx"].get<uint>();
    uint ny = this->mMetaData["grid-size"]["ny"].get<uint>();
    uint nz = this->mMetaData["grid-size"]["nz"].get<uint>();

    json properties = this->mMetaData["properties"];

    float min = 0.0f, max = 0.0f;
    if (!properties["value-range"].empty()) {
        min = properties["value-range"]["min"].get<float>();
        max = properties["value-range"]["max"].get<float>();
    }

    uint num_cracks = 1;
    if (properties["cracks"]["enable"].get<bool>()) {
        num_cracks = properties["cracks"]["count"].get<uint>();
        if (num_cracks < 1) {
            num_cracks = 1;
        }
    }

    uint num_layers = 1;
    string type_layers;
    float sample_step_layer = 0.0f;
    if (properties["layers"]["enable"].get<bool>()) {
        num_layers = properties["layers"]["count"].get<uint>();
        if (num_layers < 1) {
            num_layers = 1;
        }

        type_layers = properties["layers"]["type"].get<string>();
        if (type_layers == "smooth") {
            sample_step_layer = 1.0 * (max - min) / num_layers;
        } else if (type_layers == "sharp") {
            sample_step_layer = -1;
        }
    }

    uint layer = 1;
    float val = 1.0f;
    for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
            if (sample_step_layer == -1) {
                if (iz > (layer * (nz / num_layers))) {
                    layer++;
                    val *= 2;
                }
            } else {
                val += sample_step_layer;
            }
            for (int ix = 0; ix < nx; ++ix) {
                this->GeneratedTracesData[iz * nx + ix + (iy * nx * nz)] = val;
            }
        }
    }

}

void SyntheticModelGenerator::InjectSaltBodies() {
    json salt_bodies = this->mMetaData["properties"]["salt-bodies"];
    if (!salt_bodies["enable"].get<bool>()) {
        return;
    }

    uint nx = this->mMetaData["grid-size"]["nx"].get<uint>();
    uint ny = this->mMetaData["grid-size"]["ny"].get<uint>();
    uint nz = this->mMetaData["grid-size"]["nz"].get<uint>();

    uint salt_bodies_count = 0;
    uint salt_bodies_width = 0;
    salt_bodies_count = salt_bodies["count"].get<uint>();
    if (salt_bodies_count < 1) {
        salt_bodies_count = 1;
    }

    string s_width = salt_bodies["width"].get<string>();
    if (s_width == "narrow") {
        salt_bodies_width = 50;
    } else if (s_width == "wide") {
        salt_bodies_width = 100;
    }

    string salt_bodies_type = salt_bodies["type"].get<string>();

    random_device random;
    mt19937 generate(random());
    uniform_int_distribution<> distribution_nx(salt_bodies_width, nx - salt_bodies_width);
    uniform_int_distribution<> distribution_nz(salt_bodies_width, nz - salt_bodies_width);

    for (int is = 0; is < salt_bodies_count; ++is) {
        uint nx_start = distribution_nz(generate);
        uint nz_start = distribution_nz(generate);

        uint nx_end = nx_start + salt_bodies_width;
        uint nz_end = nz_start + salt_bodies_width;

        float val = 1;
        if (salt_bodies_type == "random") {
            val *= -10.0f;
        } else if (salt_bodies_type == "identical") {
            val = -10.0f;
        }

        for (int iy = 0; iy < ny; ++iy) {
            for (int iz = nz_start; iz < nz_end; ++iz) {
                for (int ix = nx_start; ix < nx_end; ++ix) {
                    this->GeneratedTracesData[iz * nx + ix + (iy * nx * nz)] = val;
                }
            }
        }
    }

}

void SyntheticModelGenerator::InjectCracks() {}
