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
// Created by amr-nasr on 12/12/2019.
//

#include <stbx/generators/primitive/ComputationParametersGetter.hpp>

#include <stbx/generators/primitive/ConfigurationsGenerator.hpp>

#include <operations/common/ComputationParameters.hpp>
#include <operations/common/DataTypes.h>

#include <libraries/nlohmann/json.hpp>

#include <iostream>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Logging.h"

using namespace std;
using namespace operations::common;
using namespace stbx::generators;
using json = nlohmann::json;

static std::string GetHostName()                                                                                                                        
{
    char cHostName[4096];
    gethostname(cHostName, 4096);
    return std::string(cHostName);
}


static void PrintCUDADeviceProperty(const cudaDeviceProp& prop) {
    std::cout << "\tDevice name                 : " << prop.name << std::endl;
    std::cout << "\tMemory Clock Rate (GHz)     : " << prop.memoryClockRate * 1e-6 << std::endl;
    std::cout << "\tMemory Bus Width (bits)     : " << prop.memoryBusWidth << std::endl;
    std::cout << "\tCUDA cores                  : " << prop.multiProcessorCount << std::endl;
    std::cout << "\tPeak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;
    std::cout << "\tCompute capability          : " << prop.major << "." << prop.minor << std::endl;
}

static void QueryCUDADevice()
{
    std::cout << "Querying CUDA devices on " << GetHostName() << std::endl;
    int iNumDevices(-1);
    checkCUDA(cudaGetDeviceCount(&iNumDevices));
    if (iNumDevices <= 0) {
        std::cout << "ERROR: " << GetHostName() << " does not have any available NVIDIA devices" << std::endl;
        exit(-1);
    }

    std::cout << "Number of CUDA devices found: " << iNumDevices << std::endl;
    for(int i=0; i<iNumDevices; i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "*** Device Number: " << i << " *** " << std::endl;
        PrintCUDADeviceProperty(prop);
    }
    checkCUDA(cudaSetDevice(0)); // Use first CUDA device
}


void print_parameters(ComputationParameters *parameters) {
    std::cout << endl;
    std::cout << "Used parameters : " << endl;
    std::cout << "\torder of stencil used : " << parameters->GetHalfLength() * 2 << endl;
    std::cout << "\tboundary length used : " << parameters->GetBoundaryLength() << endl;
    std::cout << "\tsource frequency : " << parameters->GetSourceFrequency() << endl;
    std::cout << "\tdt relaxation coefficient : " << parameters->GetRelaxedDT() << endl;
    std::cout << "\tblock factor in x-direction : " << parameters->GetBlockX() << endl;
    std::cout << "\tblock factor in z-direction : " << parameters->GetBlockZ() << endl;
    std::cout << "\tblock factor in y-direction : " << parameters->GetBlockY() << endl;
    std::cout << "\tUsing GPU Device - Slice z + STATIC x Hybrid" << std::endl;
    QueryCUDADevice();

    if (parameters->IsUsingWindow()) {
        std::cout << "\tWindow mode : enabled" << endl;
        if (parameters->GetLeftWindow() == 0 && parameters->GetRightWindow() == 0) {
            std::cout << "\t\tNO WINDOW IN X-axis" << endl;
        } else {
            std::cout << "\t\tLeft window : " << parameters->GetLeftWindow() << endl;
            std::cout << "\t\tRight window : " << parameters->GetRightWindow() << endl;
        }
        if (parameters->GetFrontWindow() == 0 && parameters->GetBackWindow() == 0) {
            std::cout << "\t\tNO WINDOW IN Y-axis" << endl;
        } else {
            std::cout << "\t\tFrontal window : " << parameters->GetFrontWindow() << endl;
            std::cout << "\t\tBackward window : " << parameters->GetBackWindow() << endl;
        }
        if (parameters->GetDepthWindow() != 0) {
            std::cout << "\t\tDepth window : " << parameters->GetDepthWindow() << endl;
        } else {
            std::cout << "\t\tNO WINDOW IN Z-axis" << endl;
        }
    } else {
        std::cout << "\tWindow mode : disabled (To enable set use-window=yes)..." << endl;
    }
    std::cout << endl;
}

struct Device {
    int device_name = DEF_VAL;
    string device_pattern;
};

Device ParseDevice(json &map) {
    string value = map["device"].get<std::string>();
    Device d;
    if (value != "none") {
        d.device_name = 1;
        d.device_pattern = value;
    }
    return d;
}

/*!
 * File format should be follow the following example :
 *
 * stencil-order=8
 * boundary-length=20
 * source-frequency=200
 * dt-relax=0.4
 * thread-number=4
 * block-x=500
 * block-z=44
 * block-y=5
 * Device=cpu
 */
ComputationParameters *generate_parameters(json &map) {
    std::cout << "Parsing DPC++ computation properties..." << std::endl;
    json computation_parameters_map = map["computation-parameters"];
    int boundary_length = -1, block_x = -1, block_z = -1, block_y = -1,
            order = -1;
    float dt_relax = -1, source_frequency = -1;
    uint cor_block = -1;
    HALF_LENGTH half_length = O_8;
    string device_pattern;
    ////int device_selected = -1;
    int left_win = -1, right_win = -1, front_win = -1, back_win = -1, depth_win = -1, use_window = -1, device_name = -1;
    ////SYCL_ALGORITHM selected_device = SYCL_ALGORITHM::CPU;

    auto *computationParametersGetter = new stbx::generators::ComputationParametersGetter(computation_parameters_map);
    StencilOrder so = computationParametersGetter->GetStencilOrder();
    order = so.order;
    half_length = so.half_length;

    boundary_length = computationParametersGetter->GetBoundaryLength();
    source_frequency = computationParametersGetter->GetSourceFrequency();
    dt_relax = computationParametersGetter->GetDTRelaxed();
    block_x = computationParametersGetter->GetBlock("x");
    block_y = computationParametersGetter->GetBlock("y");
    block_z = computationParametersGetter->GetBlock("z");

    Window w = computationParametersGetter->GetWindow();
    left_win = w.left_win;
    right_win = w.right_win;
    front_win = w.front_win;
    back_win = w.back_win;
    depth_win = w.depth_win;
    use_window = w.use_window;

    Device d = ParseDevice(computation_parameters_map);
    device_name = d.device_name;
    device_pattern = d.device_pattern;

    if (order == -1) {
        std::cout << "No valid value provided for key 'stencil-order'..." << std::endl;
        std::cout << "Using default stencil order of 8" << std::endl;
        half_length = O_8;
    }
    if (boundary_length == -1) {
        std::cout << "No valid value provided for key 'boundary-length'..." << std::endl;
        std::cout << "Using default boundary-length of 20" << std::endl;
        boundary_length = 20;
    }
    if (source_frequency == -1) {
        std::cout << "No valid value provided for key 'source-frequency'..."
                  << std::endl;
        std::cout << "Using default source frequency of 20" << std::endl;
        source_frequency = 20;
    }
    if (dt_relax == -1) {
        std::cout << "No valid value provided for key 'dt-relax'..." << std::endl;
        std::cout << "Using default relaxation coefficient for dt calculation of 0.4"
                  << std::endl;
        dt_relax = 0.4;
    }
    if (block_x == -1) {
        std::cout << "No valid value provided for key 'block-x'..." << std::endl;
        std::cout << "Using default blocking factor in x-direction of 560" << std::endl;
        block_x = 560;
    }
    if (block_z == -1) {
        std::cout << "No valid value provided for key 'block-z'..." << std::endl;
        std::cout << "Using default blocking factor in z-direction of 35" << std::endl;
        block_z = 35;
    }
    if (block_y == -1) {
        std::cout << "No valid value provided for key 'block-y'..." << std::endl;
        std::cout << "Using default blocking factor in y-direction of 5" << std::endl;
        block_y = 5;
    }
    if (use_window == -1) {
        std::cout << "No valid value provided for key 'use-window'..." << std::endl;
        std::cout << "Disabling window by default.." << std::endl;
        use_window = 0;
    }
    if (use_window) {
        if (left_win == -1) {
            std::cout << "No valid value provided for key 'left-window'..." << std::endl;
            std::cout
                    << "Using default window size of 0- notice if both window in an axis are 0, no windowing happens on that axis"
                    << std::endl;
            left_win = 0;
        }
        if (right_win == -1) {
            std::cout << "No valid value provided for key 'right-window'..." << std::endl;
            std::cout
                    << "Using default window size of 0- notice if both window in an axis are 0, no windowing happens on that axis"
                    << std::endl;
            right_win = 0;
        }
        if (depth_win == -1) {
            std::cout << "No valid value provided for key 'depth-window'..." << std::endl;
            std::cout << "Using default window size of 0 - notice if window is 0, no windowing happens" << std::endl;
            depth_win = 0;
        }
        if (front_win == -1) {
            std::cout << "No valid value provided for key 'front-window'..." << std::endl;
            std::cout
                    << "Using default window size of 0- notice if both window in an axis are 0, no windowing happens on that axis"
                    << std::endl;
            front_win = 0;
        }
        if (back_win == -1) {
            std::cout << "No valid value provided for key 'back-window'..." << std::endl;
            std::cout
                    << "Using default window size of 0- notice if both window in an axis are 0, no windowing happens on that axis"
                    << std::endl;
            back_win = 0;
        }
    }
    auto *parameters = new ComputationParameters(half_length);
    auto *configurationsGenerator = new ConfigurationsGenerator(map);

    /// General
    parameters->SetBoundaryLength(boundary_length);
    parameters->SetRelaxedDT(dt_relax);
    parameters->SetSourceFrequency(source_frequency);
    parameters->SetIsUsingWindow(use_window == 1);
    parameters->SetLeftWindow(left_win);
    parameters->SetRightWindow(right_win);
    parameters->SetDepthWindow(depth_win);
    parameters->SetFrontWindow(front_win);
    parameters->SetBackWindow(back_win);
    parameters->SetEquationOrder(configurationsGenerator->GetEquationOrder());
    parameters->SetApproximation(configurationsGenerator->GetApproximation());
    parameters->SetPhysics(configurationsGenerator->GetPhysics());

    /// OneAPI
    parameters->SetBlockX(block_x);
    parameters->SetBlockZ(block_z);
    parameters->SetBlockY(block_y);

    print_parameters(parameters);
    delete configurationsGenerator;
    delete computationParametersGetter;
    return parameters;
}
