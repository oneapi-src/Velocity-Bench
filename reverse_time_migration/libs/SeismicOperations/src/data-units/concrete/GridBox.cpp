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
// Created by zeyad-osama on 30/12/2020.
//

#include <operations/data-units/concrete/holders/GridBox.hpp>

using namespace operations::dataunits;
using namespace operations::exceptions;


GridBox::GridBox() {
    this->mpActualGridSize = new GridSize();
    this->mpLogicalGridSize = new GridSize();
    this->mpInitialGridSize = new GridSize();
    this->mpComputationGridSize = new GridSize();
    this->mpLogicalWindowSize = new GridSize();
    this->mpWindowProperties = new WindowProperties();
    this->mpCellDimensions = new CellDimensions();
    this->mpInitialCellDimensions = new CellDimensions();
    this->mpReferencePoint = new FPoint3D();
    this->mDT = 0.0f;
    this->mNT = 0;
}

GridBox::~GridBox() {
    delete this->mpActualGridSize;
    delete this->mpLogicalGridSize;
    delete this->mpInitialGridSize;
    delete this->mpComputationGridSize;
    delete this->mpLogicalWindowSize;
    delete this->mpWindowProperties;
    delete this->mpCellDimensions;
    delete this->mpInitialCellDimensions;
    delete this->mpReferencePoint;
};

void GridBox::SetDT(float _dt) {
    if (_dt <= 0) {
        throw IllogicalException();
    }
    this->mDT = _dt;
}

void GridBox::SetNT(float _nt) {
    if (_nt <= 0) {
        throw operations::exceptions::IllogicalException();
    }
    this->mNT = _nt;
}

void GridBox::SetReferencePoint(uint axis, float val) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    if (axis == Y_AXIS) {
        this->mpReferencePoint->y = val;
    } else if (axis == Z_AXIS) {
        this->mpReferencePoint->z = val;
    } else if (axis == X_AXIS) {
        this->mpReferencePoint->x = val;
    }
}


float GridBox::GetReferencePoint(uint axis) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    float val;
    if (axis == Y_AXIS) {
        val = this->mpReferencePoint->y;
    } else if (axis == Z_AXIS) {
        val = this->mpReferencePoint->z;
    } else if (axis == X_AXIS) {
        val = this->mpReferencePoint->x;
    }
    return val;
}


void GridBox::SetCellDimensions(uint axis, float val) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    if (axis == Y_AXIS) {
        this->mpCellDimensions->dy = val;
    } else if (axis == Z_AXIS) {
        this->mpCellDimensions->dz = val;
    } else if (axis == X_AXIS) {
        this->mpCellDimensions->dx = val;
    }
}

float GridBox::GetCellDimensions(uint axis) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    float val;
    if (axis == Y_AXIS) {
        val = this->mpCellDimensions->dy;
    } else if (axis == Z_AXIS) {
        val = this->mpCellDimensions->dz;
    } else if (axis == X_AXIS) {
        val = this->mpCellDimensions->dx;
    }
    return val;
}

void GridBox::SetInitialCellDimensions(uint axis, float val) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    if (axis == Y_AXIS) {
        this->mpInitialCellDimensions->dy = val;
    } else if (axis == Z_AXIS) {
        this->mpInitialCellDimensions->dz = val;
    } else if (axis == X_AXIS) {
        this->mpInitialCellDimensions->dx = val;
    }
}

float GridBox::GetInitialCellDimensions(uint axis) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    float val;
    if (axis == Y_AXIS) {
        val = this->mpInitialCellDimensions->dy;
    } else if (axis == Z_AXIS) {
        val = this->mpInitialCellDimensions->dz;
    } else if (axis == X_AXIS) {
        val = this->mpInitialCellDimensions->dx;
    }
    return val;
}

void GridBox::SetActualWindowSize(uint axis, uint val) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    if (axis == Y_AXIS) {
        this->mpWindowProperties->wny = val;
    } else if (axis == Z_AXIS) {
        this->mpWindowProperties->wnz = val;
    } else if (axis == X_AXIS) {
        this->mpWindowProperties->wnx = val;
    }
}

uint GridBox::GetActualWindowSize(uint axis) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    uint val;
    if (axis == Y_AXIS) {
        val = this->mpWindowProperties->wny;
    } else if (axis == Z_AXIS) {
        val = this->mpWindowProperties->wnz;
    } else if (axis == X_AXIS) {
        val = this->mpWindowProperties->wnx;
    }
    return val;
}

void GridBox::SetLogicalWindowSize(uint axis, uint val) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    if (axis == Y_AXIS) {
        this->mpLogicalWindowSize->ny = val;
    } else if (axis == Z_AXIS) {
        this->mpLogicalWindowSize->nz = val;
    } else if (axis == X_AXIS) {
        this->mpLogicalWindowSize->nx = val;
    }
}

uint GridBox::GetLogicalWindowSize(uint axis) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    uint val;
    if (axis == Y_AXIS) {
        val = this->mpLogicalWindowSize->ny;
    } else if (axis == Z_AXIS) {
        val = this->mpLogicalWindowSize->nz;
    } else if (axis == X_AXIS) {
        val = this->mpLogicalWindowSize->nx;
    }
    return val;
}

/**
 * @brief Window start per axe setter.
 * @param[in] axis      Axe direction
 * @param[in] val       Value to be set
 * @throw IllogicalException()
 */
void GridBox::SetWindowStart(uint axis, uint val) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    if (axis == Y_AXIS) {
        this->mpWindowProperties->window_start.y = val;
    } else if (axis == Z_AXIS) {
        this->mpWindowProperties->window_start.z = val;
    } else if (axis == X_AXIS) {
        this->mpWindowProperties->window_start.x = val;
    }
}

uint GridBox::GetWindowStart(uint axis) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    uint val;
    if (axis == Y_AXIS) {
        val = this->mpWindowProperties->window_start.y;
    } else if (axis == Z_AXIS) {
        val = this->mpWindowProperties->window_start.z;
    } else if (axis == X_AXIS) {
        val = this->mpWindowProperties->window_start.x;
    }
    return val;
}

void GridBox::SetActualGridSize(uint axis, uint val) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    if (axis == Y_AXIS) {
        this->mpActualGridSize->ny = val;
    } else if (axis == Z_AXIS) {
        this->mpActualGridSize->nz = val;
    } else if (axis == X_AXIS) {
        this->mpActualGridSize->nx = val;
    }
}

uint GridBox::GetActualGridSize(uint axis) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    uint val;
    if (axis == Y_AXIS) {
        val = this->mpActualGridSize->ny;
    } else if (axis == Z_AXIS) {
        val = this->mpActualGridSize->nz;
    } else if (axis == X_AXIS) {
        val = this->mpActualGridSize->nx;
    }
    return val;
}

void GridBox::SetLogicalGridSize(uint axis, uint val) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    if (axis == Y_AXIS) {
        this->mpLogicalGridSize->ny = val;
    } else if (axis == Z_AXIS) {
        this->mpLogicalGridSize->nz = val;
    } else if (axis == X_AXIS) {
        this->mpLogicalGridSize->nx = val;
    }
}

uint GridBox::GetLogicalGridSize(uint axis) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    uint val;
    if (axis == Y_AXIS) {
        val = this->mpLogicalGridSize->ny;
    } else if (axis == Z_AXIS) {
        val = this->mpLogicalGridSize->nz;
    } else if (axis == X_AXIS) {
        val = this->mpLogicalGridSize->nx;
    }
    return val;
}

void GridBox::SetComputationGridSize(uint axis, uint val) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    if (axis == Y_AXIS) {
        this->mpComputationGridSize->ny = val;
    } else if (axis == Z_AXIS) {
        this->mpComputationGridSize->nz = val;
    } else if (axis == X_AXIS) {
        this->mpComputationGridSize->nx = val;
    }
}

uint GridBox::GetComputationGridSize(uint axis) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    uint val;
    if (axis == Y_AXIS) {
        val = this->mpComputationGridSize->ny;
    } else if (axis == Z_AXIS) {
        val = this->mpComputationGridSize->nz;
    } else if (axis == X_AXIS) {
        val = this->mpComputationGridSize->nx;
    }
    return val;
}

void GridBox::SetInitialGridSize(uint axis, uint val) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    if (axis == Y_AXIS) {
        this->mpInitialGridSize->ny = val;
    } else if (axis == Z_AXIS) {
        this->mpInitialGridSize->nz = val;
    } else if (axis == X_AXIS) {
        this->mpInitialGridSize->nx = val;
    }
}

uint GridBox::GetInitialGridSize(uint axis) {
    if (is_out_of_range(axis)) {
        throw AxisException();
    }

    uint val;
    if (axis == Y_AXIS) {
        val = this->mpInitialGridSize->ny;
    } else if (axis == Z_AXIS) {
        val = this->mpInitialGridSize->nz;
    } else if (axis == X_AXIS) {
        val = this->mpInitialGridSize->nx;
    }
    return val;
}


void GridBox::RegisterWaveField(u_int16_t key, FrameBuffer<float> *ptr_wave_field) {
    key |= WAVE;
    this->mWaveFields[key] = ptr_wave_field;
}

void GridBox::RegisterParameter(u_int16_t key,
                                FrameBuffer<float> *ptr_parameter,
                                FrameBuffer<float> *ptr_parameter_window) {
    key |= PARM;
    this->mParameters[key] = ptr_parameter;

    key |= WIND;
    if (ptr_parameter_window == nullptr) {
        this->mWindowParameters[key] = ptr_parameter;
    } else {
        this->mWindowParameters[key] = ptr_parameter_window;
    }
}

void GridBox::RegisterMasterWaveField() {
    /// @todo To be implemented.
    throw NotImplementedException();
}

float *GridBox::GetMasterWaveField() {
    /// @todo To be implemented.
    throw NotImplementedException();
}

FrameBuffer<float> *GridBox::Get(u_int16_t key) {
    if (this->mWaveFields.find(key) != this->mWaveFields.end()) {
        return this->mWaveFields[key];
    } else if (this->mParameters.find(key) != this->mParameters.end()) {
        return this->mParameters[key];
    } else if (this->mWindowParameters.find(key) != this->mWindowParameters.end()) {
        return this->mWindowParameters[key];
    }
    throw NotFoundException();
}

void GridBox::Set(u_int16_t key, FrameBuffer<float> *val) {
    if (this->mWaveFields.find(key) != this->mWaveFields.end()) {
        this->mWaveFields[key] = val;
    } else if (this->mParameters.find(key) != this->mParameters.end()) {
        this->mParameters[key] = val;
    } else if (this->mWindowParameters.find(key) != this->mWindowParameters.end()) {
        this->mWindowParameters[key] = val;
    } else {
        throw NotFoundException();
    }
}

void GridBox::Set(u_int16_t key, float *val) {
    if (this->mWaveFields.find(key) != this->mWaveFields.end()) {
        this->mWaveFields[key]->SetNativePointer(val);
    } else if (this->mParameters.find(key) != this->mParameters.end()) {
        this->mParameters[key]->SetNativePointer(val);
    } else if (this->mWindowParameters.find(key) != this->mWindowParameters.end()) {
        this->mWindowParameters[key]->SetNativePointer(val);
    } else {
        throw NotFoundException();
    }
}

void GridBox::Reset(u_int16_t key) {
    this->Set(key, (FrameBuffer<float> *)
            nullptr);
}

void GridBox::Swap(u_int16_t _src, u_int16_t _dst) {
    if (this->mWaveFields.find(_src) == this->mWaveFields.end() ||
        this->mWaveFields.find(_dst) == this->mWaveFields.end()) {
        throw NotFoundException();
    }

    FrameBuffer<float> *src, *dest;
    if (this->mWaveFields.find(_src) != this->mWaveFields.end() &&
        this->mWaveFields.find(_dst) != this->mWaveFields.end()) {
        src = this->mWaveFields[_src];
        dest = this->mWaveFields[_dst];
        this->mWaveFields[_dst] = src;
        this->mWaveFields[_src] = dest;
    }
}

void GridBox::Clone(GridBox *apGridBox) {
    this->CloneMetaData(apGridBox);
    this->CloneWaveFields(apGridBox);
    this->CloneParameters(apGridBox);
}

void GridBox::CloneMetaData(GridBox *apGridBox) {
    apGridBox->SetNT(this->GetNT());
    apGridBox->SetDT(this->GetDT());

    memcpy(apGridBox->GetActualGridSize(),
           this->GetActualGridSize(),
           sizeof(GridSize));

    memcpy(apGridBox->GetLogicalGridSize(),
           this->GetLogicalGridSize(),
           sizeof(GridSize));

    memcpy(apGridBox->GetComputationGridSize(),
           this->GetComputationGridSize(),
           sizeof(GridSize));

    memcpy(apGridBox->GetLogicalWindowSize(),
           this->GetLogicalWindowSize(),
           sizeof(GridSize));

    memcpy(apGridBox->GetInitialGridSize(),
           this->GetInitialGridSize(),
           sizeof(GridSize));

    memcpy(apGridBox->GetWindowProperties(),
           this->GetWindowProperties(),
           sizeof(WindowProperties));

    memcpy(apGridBox->GetCellDimensions(),
           this->GetCellDimensions(),
           sizeof(CellDimensions));

    memcpy(apGridBox->GetInitialCellDimensions(),
           this->GetInitialCellDimensions(),
           sizeof(CellDimensions));

    memcpy(apGridBox->GetReferencePoint(),
           this->GetReferencePoint(),
           sizeof(FPoint3D));
}


void GridBox::CloneWaveFields(GridBox *apGridBox) {
    for (auto const &parameter : this->GetParameters()) {
        apGridBox->RegisterWaveField(parameter.first,
                                     parameter.second);
    }
}


void GridBox::CloneParameters(GridBox *apGridBox) {
    for (auto const &parameter : this->GetParameters()) {
        apGridBox->RegisterParameter(parameter.first,
                                     this->Get(parameter.first),
                                     this->Get(WIND | parameter.first));
    }
}


void GridBox::Report(REPORT_LEVEL aReportLevel) {
    std::cout << std::endl << "GridBox Report " << std::endl;
    std::cout << "==============================";

    uint index = 0;

    std::cout << std::endl << "Actual Grid Size: " << std::endl;
    std::cout << "- nx\t: " << GetActualGridSize(X_AXIS) << std::endl;
    std::cout << "- ny\t: " << GetActualGridSize(Y_AXIS) << std::endl;
    std::cout << "- nz\t: " << GetActualGridSize(Z_AXIS) << std::endl;
    std::cout << "- nt\t: " << GetNT() << std::endl;

    std::cout << std::endl << "Logical Grid Size: " << std::endl;
    std::cout << "- nx\t: " << GetLogicalGridSize(X_AXIS) << std::endl;
    std::cout << "- ny\t: " << GetLogicalGridSize(Y_AXIS) << std::endl;
    std::cout << "- nz\t: " << GetLogicalGridSize(Z_AXIS) << std::endl;

    std::cout << std::endl << "Actual Window Size: " << std::endl;
    std::cout << "- wnx\t: " << GetActualWindowSize(X_AXIS) << std::endl;
    std::cout << "- wny\t: " << GetActualWindowSize(Y_AXIS) << std::endl;
    std::cout << "- wnz\t: " << GetActualWindowSize(Z_AXIS) << std::endl;

    std::cout << std::endl << "Logical Window Size: " << std::endl;
    std::cout << "- wnx\t: " << GetLogicalWindowSize(X_AXIS) << std::endl;
    std::cout << "- wny\t: " << GetLogicalWindowSize(Y_AXIS) << std::endl;
    std::cout << "- wnz\t: " << GetLogicalWindowSize(Z_AXIS) << std::endl;

    std::cout << std::endl << "Computation Grid Size: " << std::endl;
    std::cout << "- x elements\t: " << GetComputationGridSize(X_AXIS) << std::endl;
    std::cout << "- y elements\t: " << GetComputationGridSize(Y_AXIS) << std::endl;
    std::cout << "- z elements\t: " << GetComputationGridSize(Z_AXIS) << std::endl;

    std::cout << std::endl << "Cell Dimensions: " << std::endl;
    std::cout << "- dx\t: " << GetCellDimensions(X_AXIS) << std::endl;
    std::cout << "- dy\t: " << GetCellDimensions(Y_AXIS) << std::endl;
    std::cout << "- dz\t: " << GetCellDimensions(Z_AXIS) << std::endl;
    std::cout << "- dt\t: " << GetDT() << std::endl;

    std::cout << std::endl << "Wave Fields: " << std::endl;
    std::cout << "- Count\t: " << GetWaveFields().size() << std::endl;
    std::cout << "- Names\t: " << std::endl;
    index = 0;
    for (auto const &wave_field : GetWaveFields()) {
        std::cout << "\t" << ++index << ". "
                  << Beautify(Stringify(wave_field.first))
                  << std::endl;
    }

    std::cout << std::endl << "Parameters: " << std::endl;
    std::cout << "- Count\t: " << GetParameters().size() << std::endl;
    std::cout << "- Names\t: " << std::endl;
    index = 0;
    for (auto const &parameter : GetParameters()) {
        std::cout << "\t" << ++index << ". "
                  << Beautify(Stringify(parameter.first))
                  << std::endl;
    }
    std::cout << std::endl;
}

bool GridBox::Has(u_int16_t key) {
    if (this->mWaveFields.find(key) != this->mWaveFields.end()) {
        return true;
    } else if (this->mParameters.find(key) != this->mParameters.end()) {
        return true;
    } else if (this->mWindowParameters.find(key) != this->mWindowParameters.end()) {
        return true;
    }
    return false;
}

void GridBox::Replace(u_int16_t *key, u_int16_t _src, u_int16_t _dest) {
    *key = ~_src & *key;
    SetBits(key, _dest);
}

/**
 * @brief Converts a given key (i.e. u_int16_t) to string.
 * @param[in] key
 * @return[out] String value
 */
std::string GridBox::Stringify(u_int16_t key) {
    std::string str = "";

    /// Window
    if (Includes(key, WIND)) {
        str += "window_";
    }

    /// Wave Field / Parameter
    if (Includes(key, WAVE)) {
        str += "wave_";

        /// Pressure / Particle
        if (Includes(key, GB_PRSS)) {
            str += "pressure_";
        } else if (Includes(key, GB_PRTC)) {
            str += "particle_";
        }
    } else if (Includes(key, PARM)) {
        str += "parameter_";

        /// Parameter
        if (Includes(key, GB_DEN)) {
            str += "density_";
        } else if (Includes(key, GB_PHI)) {
            str += "phi_";
        } else if (Includes(key, GB_THT)) {
            str += "theta_";
        } else if (Includes(key, GB_EPS)) {
            str += "epsilon_";
        } else if (Includes(key, GB_DLT)) {
            str += "delta_";
        } else if (Includes(key, GB_VEL)) {
            str += "velocity_";
        }
    }

    /// Time
    if (Includes(key, PREV)) {
        str += "prev_";
    } else if (Includes(key, NEXT)) {
        str += "next_";
    } else if (Includes(key, CURR)) {
        str += "curr_";
    }

    /// Direction
    if (Includes(key, DIR_X)) {
        str += "x_";
    } else if (Includes(key, DIR_Y)) {
        str += "y_";
    } else if (Includes(key, DIR_Z)) {
        str += "z_";
    }
    return str;
}

std::string GridBox::Beautify(std::string const &str) {
    return this->Capitalize(this->ReplaceAll(str, "_", " "));
}

std::string GridBox::Capitalize(std::string str) {
    bool cap = true;
    for (unsigned int i = 0; i <= str.length(); i++) {
        if (isalpha(str[i]) && cap == true) {
            str[i] = toupper(str[i]);
            cap = false;
        } else if (isspace(str[i])) {
            cap = true;
        }
    }
    return str;
}

std::string GridBox::ReplaceAll(std::string str,
                                const std::string &from,
                                const std::string &to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}
