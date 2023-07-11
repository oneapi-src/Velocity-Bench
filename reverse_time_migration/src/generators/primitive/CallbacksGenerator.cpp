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


// Created by marwan on 08/02/2021.
//

#include <stbx/generators/primitive/CallbacksGenerator.hpp>

#include <stbx/generators/common/Keys.hpp>
#include <operations/helpers/callbacks/primitive/CallbackCollection.hpp>
#include <operations/helpers/callbacks/concrete/CSVWriter.h>
#include <operations/helpers/callbacks/concrete/ImageWriter.h>
#include <operations/helpers/callbacks/concrete/NormWriter.h>
#include <operations/helpers/callbacks/concrete/SegyWriter.h>
#include <operations/helpers/callbacks/concrete/SUWriter.h>
#include <operations/helpers/callbacks/concrete/BinaryWriter.h>

#include <iostream>
#include <string>

using namespace stbx::generators;
using namespace operations::helpers::callbacks;


CallbacksGenerator::CallbacksGenerator(const std::string &aWritePath, nlohmann::json &aMap) {
    this->mMap = aMap;
    this->mWritePath = aWritePath;
    this->mpCollection = new CallbackCollection();
}

CallbacksGenerator::~CallbacksGenerator()
{
    delete mpCollection;
}

void CallbacksGenerator::GenerateCallbacks() {
    GetImageCallback();
    GetNormCallback();
    GetCsvCallback();
    GetSegyCallback();
    GetSuCallback();
    GetBinaryCallback();
}

CallbacksGenerator::WritersBooleans CallbacksGenerator::GenerateWriters() {
    struct CallbacksGenerator::WritersBooleans w;
    nlohmann::json map = this->mMap[K_WRITERS];
    if (map[K_PARAMETERS][K_ENABLE].get<bool>()) {
        w.WriteParams = true;
        if (!map[K_PARAMETERS][K_OUTPUT].is_null()) {
            w.VecParams = map[K_PARAMETERS][K_OUTPUT].get<std::vector<std::string>>();
        }
    }
    if (map[K_FORWARD][K_ENABLE].get<bool>()) {
        w.WriteForward = true;
    }
    if (map[K_BACKWARD][K_ENABLE].get<bool>()) {
        w.WriteBackward = true;
    }
    if (map[K_REVERSE][K_ENABLE].get<bool>()) {
        w.WriteReverse = true;
    }
    if (map[K_MIGRATION][K_ENABLE].get<bool>()) {
        w.WriteMigration = true;
    }
    if (map[K_RE_EXTENDED_PARAMETERS][K_ENABLE].get<bool>()) {
        w.WriteReExtendedParams = true;
        if (!map[K_RE_EXTENDED_PARAMETERS][K_OUTPUT].is_null()) {
            w.VecReExtendedParams = map[K_RE_EXTENDED_PARAMETERS][K_OUTPUT].get<std::vector<std::string>>();
        }
    }
    if (map[K_SINGLE_SHOT_CORR][K_ENABLE].get<bool>()) {
        w.WriteSingleShotCorrelation = true;
    }
    if (map[K_EACH_STACKED_SHOT][K_ENABLE].get<bool>()) {
        w.WriteEachStackedShot = true;
    }
    if (map[K_TRACE_RAW][K_ENABLE].get<bool>()) {
        w.WriteTracesRaw = true;
    }
    if (map[K_TRACES_PREPROCESSED][K_ENABLE].get<bool>()) {
        w.WriteTracesPreprocessed = true;
    }
    return w;
}

void CallbacksGenerator::GetImageCallback() {
#ifdef USE_OpenCV
    if (this->mMap[K_IMAGE][K_ENABLE].get<bool>()) {
        int show_each = 200;
        float percentile = 98.5;

        if (!this->mMap[K_IMAGE][K_SHOW_EACH].is_null()) {
            show_each = this->mMap[K_IMAGE][K_SHOW_EACH].get<int>();
        }
        if (!this->mMap[K_IMAGE][K_PERC].is_null()) {
            percentile = this->mMap[K_IMAGE][K_PERC].get<float>();
        }
        CallbacksGenerator::WritersBooleans w = this->GenerateWriters();
        this->mpCollection->RegisterCallback(new ImageWriter(show_each,
                                                             w.WriteParams,
                                                             w.WriteForward,
                                                             w.WriteBackward,
                                                             w.WriteReverse,
                                                             w.WriteMigration,
                                                             w.WriteReExtendedParams,
                                                             w.WriteSingleShotCorrelation,
                                                             w.WriteEachStackedShot,
                                                             w.WriteTracesRaw,
                                                             w.WriteTracesPreprocessed,
                                                             w.VecParams,
                                                             w.VecReExtendedParams,
                                                             percentile,
                                                             this->mWritePath));
        std::cout << "Creating image callback with show_each = " << show_each
                  << " and percentile = " << percentile << std::endl;
    }
#endif
}

void CallbacksGenerator::GetCsvCallback() {
    if (this->mMap[K_CSV][K_ENABLE].get<bool>()) {
        int show_each = 200;

        if (!this->mMap[K_CSV][K_SHOW_EACH].is_null()) {
            show_each = this->mMap[K_CSV][K_SHOW_EACH].get<int>();
        }

        CallbacksGenerator::WritersBooleans w = this->GenerateWriters();
        this->mpCollection->RegisterCallback(new CsvWriter(show_each,
                                                           w.WriteParams,
                                                           w.WriteForward,
                                                           w.WriteBackward,
                                                           w.WriteReverse,
                                                           w.WriteMigration,
                                                           w.WriteReExtendedParams,
                                                           w.WriteSingleShotCorrelation,
                                                           w.WriteEachStackedShot,
                                                           w.WriteTracesRaw,
                                                           w.WriteTracesPreprocessed,
                                                           w.VecParams,
                                                           w.VecReExtendedParams,
                                                           this->mWritePath));
        std::cout << "Creating CSV callback with show_each = " << show_each << std::endl;
    }
}

void CallbacksGenerator::GetNormCallback() {
    if (this->mMap[K_NORM][K_ENABLE].get<bool>()) {
        int show_each = 200;
        if (!this->mMap[K_NORM][K_SHOW_EACH].is_null()) {
            show_each = this->mMap[K_NORM][K_SHOW_EACH].get<int>();
        }
        this->mpCollection->RegisterCallback(new NormWriter(show_each,
                                                            true,
                                                            true,
                                                            true,
                                                            this->mWritePath));
        std::cout << "Creating norm callback with show_each = " << show_each << std::endl;
    }
}

void CallbacksGenerator::GetSegyCallback() {
    if (this->mMap[K_SEGY][K_ENABLE].get<bool>()) {
        int show_each = 200;

        if (!this->mMap[K_SEGY][K_SHOW_EACH].is_null()) {
            show_each = this->mMap[K_SEGY][K_SHOW_EACH].get<int>();
        }
        CallbacksGenerator::WritersBooleans w = this->GenerateWriters();
        this->mpCollection->RegisterCallback(new SegyWriter(show_each,
                                                            w.WriteParams,
                                                            w.WriteForward,
                                                            w.WriteBackward,
                                                            w.WriteReverse,
                                                            w.WriteMigration,
                                                            w.WriteReExtendedParams,
                                                            w.WriteSingleShotCorrelation,
                                                            w.WriteEachStackedShot,
                                                            w.WriteTracesRaw,
                                                            w.WriteTracesPreprocessed,
                                                            w.VecParams,
                                                            w.VecReExtendedParams,
                                                            this->mWritePath));
        std::cout << "Creating SEG-Y callback with show_each = " << show_each << std::endl;
    }
}

void CallbacksGenerator::GetSuCallback() {
    if (this->mMap[K_SU][K_ENABLE].get<bool>()) {
        int show_each = 200;
        bool write_little_endian = false;

        if (!this->mMap[K_SU][K_SHOW_EACH].is_null()) {
            show_each = this->mMap[K_SU][K_SHOW_EACH].get<int>();
        }
        if (this->mMap[K_SU][K_LITTLE_ENDIAN].get<bool>()) {
            write_little_endian = true;
        }
        CallbacksGenerator::WritersBooleans w = this->GenerateWriters();
        auto *su_writer =
                new SuWriter(show_each,
                             w.WriteParams,
                             w.WriteForward,
                             w.WriteBackward,
                             w.WriteReverse,
                             w.WriteMigration,
                             w.WriteReExtendedParams,
                             w.WriteSingleShotCorrelation,
                             w.WriteEachStackedShot,
                             w.WriteTracesRaw,
                             w.WriteTracesPreprocessed,
                             w.VecParams,
                             w.VecReExtendedParams,
                             this->mWritePath,
                             write_little_endian);
        this->mpCollection->RegisterCallback(su_writer);
        if (write_little_endian) {
            std::cout << "Creating SU callback in little endian format with show_each = " << show_each << std::endl;
        } else {
            std::cout << "Creating SU callback in big endian format with show_each = " << show_each << std::endl;
        }
    }
}

void CallbacksGenerator::GetBinaryCallback() {
    if (this->mMap[K_BIN][K_ENABLE].get<bool>()) {
        int show_each = 200;

        if (!this->mMap[K_BIN][K_SHOW_EACH].is_null()) {
            show_each = this->mMap[K_BIN][K_SHOW_EACH].get<int>();
        }
        CallbacksGenerator::WritersBooleans w = this->GenerateWriters();
        auto *binary_writer =
                new BinaryWriter(show_each,
                                 w.WriteParams,
                                 w.WriteForward,
                                 w.WriteBackward,
                                 w.WriteReverse,
                                 w.WriteMigration,
                                 w.WriteReExtendedParams,
                                 w.WriteSingleShotCorrelation,
                                 w.WriteEachStackedShot,
                                 w.WriteTracesRaw,
                                 w.WriteTracesPreprocessed,
                                 w.VecParams,
                                 w.VecReExtendedParams,
                                 this->mWritePath);
        this->mpCollection->RegisterCallback(binary_writer);
        std::cout << "Creating binary callback with show_each = " << show_each << std::endl;
    }
}
