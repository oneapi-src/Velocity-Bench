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
// Created by amr-nasr on 12/11/2019.
//

#include <operations/helpers/callbacks/interface/WriterCallback.h>

#include <operations/common/DataTypes.h>

#include <map>
#include <vector>
#include <sys/stat.h>

#define CAT_STR_TO_CHR(a, b) ((char *)string(a + b).c_str())

using namespace std;
using namespace operations::helpers::callbacks;
using namespace operations::common;
using namespace operations::dataunits;


/// Helper functions to be relocated/replaced
/// after further implementations
/// {
bool is_path_exist(const std::string &s) {
    struct stat buffer;
    return (stat(s.c_str(), &buffer) == 0);
}

u_int16_t get_callbacks_map(const string &key) {
    /// Initialized upon first call to the function.
    static std::map<std::string, u_int16_t> callbacks_params_map = {
            {"velocity", PARM | GB_VEL},
            {"delta",    PARM | GB_DLT},
            {"epsilon",  PARM | GB_EPS},
            {"theta",    PARM | GB_THT},
            {"phi",      PARM | GB_PHI},
            {"density",  PARM | GB_DEN}
    };
    return callbacks_params_map.at(key);
}

std::map<std::string, u_int16_t> get_params_map() {
    /// Initialized upon first call to the function.
    static std::map<std::string, u_int16_t> callbacks_params_map = {
            {"velocity", PARM | GB_VEL},
            {"delta",    PARM | GB_DLT},
            {"epsilon",  PARM | GB_EPS},
            {"theta",    PARM | GB_THT},
            {"phi",      PARM | GB_PHI},
            {"density",  PARM | GB_DEN}
    };
    return std::move(callbacks_params_map);
}

std::map<std::string, u_int16_t> get_pressure_map() {
    // Initialized upon the first call to the function
    static std::map<std::string, u_int16_t> callbacks_pressure_map = {
            {"vertical",   WAVE | GB_PRSS | CURR | DIR_Z},
            {"horizontal", WAVE | GB_PRSS | CURR | DIR_X}
    };
    return callbacks_pressure_map;
}
/// }
/// End of helper functions


float *Unpad(const float *apOriginalArray, uint nx, uint ny, uint nz,
             uint nx_original, uint ny_original, uint nz_original) {
    if (nx == nx_original && nz == nz_original && ny == ny_original) {
        return nullptr;
    } else {
        auto copy_array = new float[ny_original * nz_original * nx_original];
        for (uint iy = 0; iy < ny_original; iy++) {
            for (uint iz = 0; iz < nz_original; iz++) {
                for (uint ix = 0; ix < nx_original; ix++) {
                    copy_array[iy * nz_original * nx_original + iz * nx_original + ix] =
                            apOriginalArray[iy * nz * nx + iz * nx + ix];
                }
            }
        }
        return copy_array;
    }
}

WriterCallback::WriterCallback(uint show_each, bool write_params, bool write_forward,
                               bool write_backward, bool write_reverse,
                               bool write_migration, bool write_re_extended_params,
                               bool write_single_shot_correlation,
                               bool write_each_stacked_shot, bool write_traces_raw,
                               bool write_traces_preprocessed,
                               const std::vector<std::string> &vec_params,
                               const std::vector<std::string> &vec_re_extended_params,
                               const string &write_path, const string &folder_name) {
    this->mShowEach = show_each;
    this->mShotCount = 0;
    this->mIsWriteParams = write_params;
    this->mIsWriteForward = write_forward;
    this->mIsWriteBackward = write_backward;
    this->mIsWriteReverse = write_reverse;
    this->mIsWriteMigration = write_migration;
    this->mIsWriteReExtendedParams = write_re_extended_params;
    this->mIsWriteSingleShotCorrelation = write_single_shot_correlation;
    this->mIsWriteEachStackedShot = write_each_stacked_shot;
    this->mIsWriteTracesRaw = write_traces_raw;
    this->mIsWriteTracesPreprocessed = write_traces_preprocessed;
    this->mParamsVec = vec_params;
    this->mReExtendedParamsVec = vec_re_extended_params;

    if (this->mParamsVec.empty()) {
        for (auto const &pair : get_params_map()) {
            this->mParamsVec.push_back(pair.first);
        }
    }
    if (this->mReExtendedParamsVec.empty()) {
        for (auto const &pair: get_params_map()) {
            this->mReExtendedParamsVec.push_back(pair.first);
        }
    }

    this->mWritePath = write_path;
    mkdir(write_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    this->mWritePath = this->mWritePath + "/" + folder_name;
    mkdir(this->mWritePath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    if (write_re_extended_params) {
        mkdir(CAT_STR_TO_CHR(this->mWritePath, "/parameters"),
              S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    if (write_single_shot_correlation) {
        mkdir(CAT_STR_TO_CHR(this->mWritePath, "/shots"),
              S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    if (write_each_stacked_shot) {
        mkdir(CAT_STR_TO_CHR(this->mWritePath, "/stacked_shots"),
              S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    if (write_forward) {
        mkdir(CAT_STR_TO_CHR(this->mWritePath, "/forward"),
              S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    if (write_reverse) {
        mkdir(CAT_STR_TO_CHR(this->mWritePath, "/reverse"),
              S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    if (write_backward) {
        mkdir(CAT_STR_TO_CHR(this->mWritePath, "/backward"),
              S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    if (write_traces_raw) {
        mkdir(CAT_STR_TO_CHR(this->mWritePath, "/traces_raw"),
              S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    if (write_traces_preprocessed) {
        mkdir(CAT_STR_TO_CHR(this->mWritePath, "/traces"),
              S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
}

void WriterCallback::BeforeInitialization(ComputationParameters *apParameters) {}

void WriterCallback::AfterInitialization(GridBox *apGridBox) {
    if (this->mIsWriteParams) {
        uint pnx = apGridBox->GetActualGridSize(X_AXIS);
        uint pny = apGridBox->GetActualGridSize(Y_AXIS);
        uint pnz = apGridBox->GetActualGridSize(Z_AXIS);
        uint nx = apGridBox->GetLogicalGridSize(X_AXIS);
        uint ny = apGridBox->GetLogicalGridSize(Y_AXIS);
        uint nz = apGridBox->GetLogicalGridSize(Z_AXIS);

        float dx = apGridBox->GetCellDimensions(X_AXIS);
        float dy = apGridBox->GetCellDimensions(Y_AXIS);
        float dz = apGridBox->GetCellDimensions(Z_AXIS);
        float dt = apGridBox->GetDT();

        for (const auto &param : this->mParamsVec) {
            if (apGridBox->Has(get_callbacks_map(param))) {
                float *arr = apGridBox->Get(get_callbacks_map(param))->GetHostPointer();
                float *unpadded_arr = Unpad(arr, pnx, pny, pnz, nx, ny, nz);
                if (unpadded_arr) {
                    arr = unpadded_arr;
                }

                uint nt = apGridBox->GetNT();

                WriteResult(nx, ny, nz, nt, dx, dy, dz, dt, arr,
                            CAT_STR_TO_CHR(this->mWritePath, "/" + param + this->GetExtension()), false);
                delete [] unpadded_arr;
            }
        }
    }
}

void WriterCallback::BeforeShotPreprocessing(TracesHolder *traces) {
    if (this->mIsWriteTracesRaw) {
        uint nz = traces->SampleNT;
        uint nx = traces->ReceiversCountX;
        uint ny = traces->ReceiversCountY;

        uint nt = traces->SampleNT;
        float dt = traces->SampleDT;

        float dx = 0.0;
        float dy = 0.0;
        float dz = 0.0;

        WriteResult(nx, ny, nz, nt, dx, dy, dz, dt, traces->Traces,
                    (char *) string(this->mWritePath + "/traces_raw/trace_" +
                                    to_string(this->mShotCount) + this->GetExtension())
                            .c_str(),
                    true);
    }
}

void WriterCallback::AfterShotPreprocessing(TracesHolder *traces) {
    if (this->mIsWriteTracesPreprocessed) {
        uint nz = traces->SampleNT;
        uint nx = traces->ReceiversCountX;
        uint ny = traces->ReceiversCountY;

        uint nt = traces->SampleNT;
        float dt = traces->SampleDT;

        float dx = 0.0;
        float dy = 0.0;
        float dz = 0.0;

        WriteResult(nx, ny, nz, nt, dx, dy, dz, dt, traces->Traces,
                    (char *) string(this->mWritePath + "/traces/trace_" +
                                    to_string(this->mShotCount) + this->GetExtension())
                            .c_str(),
                    true);
    }
}

void WriterCallback::BeforeForwardPropagation(GridBox *apGridBox) {
    if (this->mIsWriteReExtendedParams) {
        uint pwnx = apGridBox->GetActualWindowSize(X_AXIS);
        uint pwny = apGridBox->GetActualWindowSize(Y_AXIS);
        uint pwnz = apGridBox->GetActualWindowSize(Z_AXIS);
        uint wnx = apGridBox->GetLogicalWindowSize(X_AXIS);
        uint wny = apGridBox->GetLogicalWindowSize(Y_AXIS);
        uint wnz = apGridBox->GetLogicalWindowSize(Z_AXIS);
        uint nt = apGridBox->GetNT();

        float dx = apGridBox->GetCellDimensions(X_AXIS);
        float dy = apGridBox->GetCellDimensions(Y_AXIS);
        float dz = apGridBox->GetCellDimensions(Z_AXIS);
        float dt = apGridBox->GetDT();

        for (const auto &param : this->mReExtendedParamsVec) {
            if (apGridBox->Has(get_callbacks_map(param))) {
                float *arr = apGridBox->Get(get_callbacks_map(param) | WIND)->GetHostPointer();
                float *unpadded_arr = Unpad(arr, pwnx, pwny, pwnz, wnx, wny, wnz);
                if (unpadded_arr) {
                    arr = unpadded_arr;
                }
                WriteResult(wnx, wny, wnz, nt, dx, dy, dz, dt, arr,
                            (char *) string(this->mWritePath + "/parameters/" + param + "_" +
                                            to_string(this->mShotCount) + this->GetExtension())
                                    .c_str(),
                            false);
                delete [] unpadded_arr;
            }
        }
    }
}


void WriterCallback::AfterForwardStep(GridBox *apGridBox, uint time_step) {
    if (mIsWriteForward && time_step % mShowEach == 0) {
        uint pwnx = apGridBox->GetActualWindowSize(X_AXIS);
        uint pwny = apGridBox->GetActualWindowSize(Y_AXIS);
        uint pwnz = apGridBox->GetActualWindowSize(Z_AXIS);
        uint wnx = apGridBox->GetLogicalWindowSize(X_AXIS);
        uint wny = apGridBox->GetLogicalWindowSize(Y_AXIS);
        uint wnz = apGridBox->GetLogicalWindowSize(Z_AXIS);
        uint nt = apGridBox->GetNT();

        float dx = apGridBox->GetCellDimensions(X_AXIS);
        float dy = apGridBox->GetCellDimensions(Y_AXIS);
        float dz = apGridBox->GetCellDimensions(Z_AXIS);
        float dt = apGridBox->GetDT();

        for (const auto &pressure : get_pressure_map()) {
            if (apGridBox->Has(pressure.second)) {
                if (!is_path_exist(string(this->mWritePath + "/forward/" + pressure.first)))
                    mkdir(CAT_STR_TO_CHR(this->mWritePath + "/forward/", pressure.first),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

                float *arr = apGridBox->Get(pressure.second)->GetHostPointer();
                float *unpadded_arr = Unpad(arr, pwnx, pwny, pwnz, wnx, wny, wnz);
                if (unpadded_arr) {
                    arr = unpadded_arr;
                }

                WriteResult(wnx, wny, wnz, nt, dx, dy, dz, dt,
                            arr,
                            (char *) string(this->mWritePath + "/forward/" + pressure.first
                                            + "/forward_" + pressure.first + "_" +
                                            to_string(time_step) + this->GetExtension()).c_str(),
                            false);
                delete [] unpadded_arr;
            }
        }
    }
}

void WriterCallback::BeforeBackwardPropagation(GridBox *apGridBox) {
    if (mIsWriteReExtendedParams) {
        uint pwnx = apGridBox->GetActualWindowSize(X_AXIS);
        uint pwny = apGridBox->GetActualWindowSize(Y_AXIS);
        uint pwnz = apGridBox->GetActualWindowSize(Z_AXIS);
        uint wnx = apGridBox->GetLogicalWindowSize(X_AXIS);
        uint wny = apGridBox->GetLogicalWindowSize(Y_AXIS);
        uint wnz = apGridBox->GetLogicalWindowSize(Z_AXIS);
        uint nt = apGridBox->GetNT();

        float dx = apGridBox->GetCellDimensions(X_AXIS);
        float dy = apGridBox->GetCellDimensions(Y_AXIS);
        float dz = apGridBox->GetCellDimensions(Z_AXIS);
        float dt = apGridBox->GetDT();

        for (const auto &param : this->mReExtendedParamsVec) {
            if (apGridBox->Has(get_callbacks_map(param))) {
                float *arr = apGridBox->Get(get_callbacks_map(param) | WIND)->GetHostPointer();
                float *unpadded_arr = Unpad(arr, pwnx, pwny, pwnz, wnx, wny, wnz);
                if (unpadded_arr) {
                    arr = unpadded_arr;
                }

                WriteResult(wnx, wny, wnz, nt, dx, dy, dz, dt, arr,
                            (char *) string(this->mWritePath + "/parameters/" + param + "_backward_" +
                                            to_string(mShotCount) + this->GetExtension())
                                    .c_str(),
                            false);
                delete [] unpadded_arr;
            }
        }
    }
}

void WriterCallback::AfterBackwardStep(
        GridBox *apGridBox, uint time_step) {
    if (mIsWriteBackward && time_step % mShowEach == 0) {
        uint pwnx = apGridBox->GetActualWindowSize(X_AXIS);
        uint pwny = apGridBox->GetActualWindowSize(Y_AXIS);
        uint pwnz = apGridBox->GetActualWindowSize(Z_AXIS);
        uint wnx = apGridBox->GetLogicalWindowSize(X_AXIS);
        uint wny = apGridBox->GetLogicalWindowSize(Y_AXIS);
        uint wnz = apGridBox->GetLogicalWindowSize(Z_AXIS);
        uint nt = apGridBox->GetNT();

        float dx = apGridBox->GetCellDimensions(X_AXIS);
        float dy = apGridBox->GetCellDimensions(Y_AXIS);
        float dz = apGridBox->GetCellDimensions(Z_AXIS);
        float dt = apGridBox->GetDT();

        for (const auto &pressure : get_pressure_map()) {
            if (apGridBox->Has(pressure.second)) {
                if (!is_path_exist(string(this->mWritePath + "/backward/" + pressure.first)))
                    mkdir(CAT_STR_TO_CHR(this->mWritePath + "/backward/", pressure.first),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

                float *arr = apGridBox->Get(pressure.second)->GetHostPointer();
                float *unpadded_arr = Unpad(arr, pwnx, pwny, pwnz, wnx, wny, wnz);
                if (unpadded_arr) {
                    arr = unpadded_arr;
                }

                WriteResult(wnx, wny, wnz, nt, dx, dy, dz, dt,
                            arr,
                            (char *) string(this->mWritePath + "/backward/" + pressure.first
                                            + "/backward_" + pressure.first + "_" +
                                            to_string(time_step) + this->GetExtension()).c_str(),
                            false);
                delete [] unpadded_arr;
            }
        }
    }
}

void WriterCallback::AfterFetchStep(
        GridBox *apGridBox, uint time_step) {
    if (mIsWriteReverse && time_step % mShowEach == 0) {
        uint pwnx = apGridBox->GetActualWindowSize(X_AXIS);
        uint pwny = apGridBox->GetActualWindowSize(Y_AXIS);
        uint pwnz = apGridBox->GetActualWindowSize(Z_AXIS);
        uint wnx = apGridBox->GetLogicalWindowSize(X_AXIS);
        uint wny = apGridBox->GetLogicalWindowSize(Y_AXIS);
        uint wnz = apGridBox->GetLogicalWindowSize(Z_AXIS);
        uint nt = apGridBox->GetNT();

        float dx = apGridBox->GetCellDimensions(X_AXIS);
        float dy = apGridBox->GetCellDimensions(Y_AXIS);
        float dz = apGridBox->GetCellDimensions(Z_AXIS);
        float dt = apGridBox->GetDT();

        for (const auto &pressure : get_pressure_map()) {
            if (apGridBox->Has(pressure.second)) {
                if (!is_path_exist(string(this->mWritePath + "/reverse/" + pressure.first)))
                    mkdir(CAT_STR_TO_CHR(this->mWritePath + "/reverse/", pressure.first),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);


                float *arr = apGridBox->Get(pressure.second)->GetHostPointer();
                float *unpadded_arr = Unpad(arr, pwnx, pwny, pwnz, wnx, wny, wnz);
                if (unpadded_arr) {
                    arr = unpadded_arr;
                }

                WriteResult(wnx, wny, wnz, nt, dx, dy, dz, dt,
                            arr,
                            (char *) string(this->mWritePath + "/reverse/" + pressure.first
                                            + "/reverse_" + pressure.first + "_" +
                                            to_string(time_step) + this->GetExtension()).c_str(),
                            false);
                delete [] unpadded_arr;
            }
        }
    }
}

void WriterCallback::BeforeShotStacking(
        GridBox *apGridBox, FrameBuffer<float> *shot_correlation) {
    if (mIsWriteSingleShotCorrelation) {
        uint pwnx = apGridBox->GetActualWindowSize(X_AXIS);
        uint pwny = apGridBox->GetActualWindowSize(Y_AXIS);
        uint pwnz = apGridBox->GetActualWindowSize(Z_AXIS);
        uint wnx = apGridBox->GetLogicalWindowSize(X_AXIS);
        uint wny = apGridBox->GetLogicalWindowSize(Y_AXIS);
        uint wnz = apGridBox->GetLogicalWindowSize(Z_AXIS);
        uint nt = apGridBox->GetNT();

        float *arr = shot_correlation->GetHostPointer();
        float *unpadded_arr = Unpad(arr, pwnx, pwny, pwnz, wnx, wny, wnz);
        if (unpadded_arr) {
            arr = unpadded_arr;
        }

        float dx = apGridBox->GetCellDimensions(X_AXIS);
        float dy = apGridBox->GetCellDimensions(Y_AXIS);
        float dz = apGridBox->GetCellDimensions(Z_AXIS);
        float dt = apGridBox->GetDT();

        WriteResult(wnx, wny, wnz, nt, dx, dy, dz, dt,
                    arr,
                    (char *) string(this->mWritePath + "/shots/correlation_" +
                                    to_string(mShotCount) + this->GetExtension())
                            .c_str(),
                    false);
        delete [] unpadded_arr;
    }
}

void WriterCallback::AfterShotStacking(
        GridBox *apGridBox, FrameBuffer<float> *stacked_shot_correlation) {
    if (mIsWriteEachStackedShot) {
        uint pnx = apGridBox->GetActualGridSize(X_AXIS);
        uint pny = apGridBox->GetActualGridSize(Y_AXIS);
        uint pnz = apGridBox->GetActualGridSize(Z_AXIS);
        uint nx = apGridBox->GetLogicalGridSize(X_AXIS);
        uint ny = apGridBox->GetLogicalGridSize(Y_AXIS);
        uint nz = apGridBox->GetLogicalGridSize(Z_AXIS);
        uint nt = apGridBox->GetNT();

        float *arr = stacked_shot_correlation->GetHostPointer();
        float *unpadded_arr = Unpad(arr, pnx, pny, pnz, nx, ny, nz);
        if (unpadded_arr) {
            arr = unpadded_arr;
        }

        float dx = apGridBox->GetCellDimensions(X_AXIS);
        float dy = apGridBox->GetCellDimensions(Y_AXIS);
        float dz = apGridBox->GetCellDimensions(Z_AXIS);
        float dt = apGridBox->GetDT();

        WriteResult(nx, ny, nz, nt, dx, dy, dz, dt,
                    arr,
                    (char *) string(this->mWritePath +
                                    "/stacked_shots/stacked_correlation_" +
                                    to_string(mShotCount) + this->GetExtension())
                            .c_str(),
                    false);
        delete [] unpadded_arr;
    }

    this->mShotCount++;
}

void WriterCallback::AfterMigration(
        GridBox *apGridBox, FrameBuffer<float> *stacked_shot_correlation) {
    if (mIsWriteMigration) {
        uint pnx = apGridBox->GetActualGridSize(X_AXIS);
        uint pny = apGridBox->GetActualGridSize(Y_AXIS);
        uint pnz = apGridBox->GetActualGridSize(Z_AXIS);
        uint nx = apGridBox->GetLogicalGridSize(X_AXIS);
        uint ny = apGridBox->GetLogicalGridSize(Y_AXIS);
        uint nz = apGridBox->GetLogicalGridSize(Z_AXIS);
        uint nt = apGridBox->GetNT();

        float *arr = stacked_shot_correlation->GetHostPointer();
        float *unpadded_arr = Unpad(arr, pnx, pny, pnz, nx, ny, nz);
        if (unpadded_arr) {
            arr = unpadded_arr;
        }

        float dx = apGridBox->GetCellDimensions(X_AXIS);
        float dy = apGridBox->GetCellDimensions(Y_AXIS);
        float dz = apGridBox->GetCellDimensions(Z_AXIS);
        float dt = apGridBox->GetDT();

        WriteResult(nx, ny, nz, nt, dx, dy, dz, dt,
                    arr,
                    CAT_STR_TO_CHR(this->mWritePath, "/migration" + this->GetExtension()), false);
        delete [] unpadded_arr;
    }
}
